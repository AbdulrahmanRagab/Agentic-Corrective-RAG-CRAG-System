import streamlit as st
import os
import tempfile
from typing import List, Annotated, Literal, TypedDict, Sequence
from pydantic import BaseModel, Field

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document


# LangGraph
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --- Page Configuration ---
st.set_page_config(page_title="Corrective RAG Assistant", layout="wide")

# --- UI: Sidebar Configuration ---
st.sidebar.header("Configuration Panel")

# 1. AI Provider Selection (Added OpenRouter)
provider = st.sidebar.selectbox(
    "AI Provider",
    ["OpenAI", "Groq", "OpenRouter"]
)

api_key = st.sidebar.text_input(f"{provider} API Key", type="password")

# 2. Model Selection based on Provider
if provider == "OpenAI":
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
elif provider == "Groq":
    model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
elif provider == "OpenRouter":
    # Common OpenRouter models (you can add more)
    model_options = [
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "mistralai/mistral-7b-instruct"
    ]
else:
    model_options = ["gpt-4o"]

selected_model = st.sidebar.selectbox("Model", model_options)

# 3. Tavily API Key (Required for Project B's Web Search logic)
tavily_api_key = st.sidebar.text_input("Tavily API Key (for Web Search)", type="password")
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

# 4. Token Control
max_tokens = st.sidebar.number_input("Max Response Tokens", min_value=50, max_value=4096, value=500, step=50)
show_steps = st.sidebar.toggle("Show Agent Steps", value=True)

st.sidebar.markdown("---")

# 5. PDF Upload Logic
st.sidebar.header("Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

# 6. Embed Button (Process only on click)
embed_clicked = st.sidebar.button("Embed Document")

if st.sidebar.button("Clear Chat / Reset"):
    st.session_state.messages = []
    st.session_state.graph = None
    st.rerun()

# --- Helper Functions ---

@st.cache_resource
def get_embedding_model():
    """Loads the embedding model once."""
    # Using a standard lightweight model for demo purposes. 
    # You can switch back to local paths or 'google/embeddinggemma-300m' if you have the resources.
    return HuggingFaceEmbeddings(model_name=r"C:\Users\Abdelrahman\.cache\huggingface\hub\models--google--embeddinggemma-300m\snapshots\64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2")

def build_vectorstore(uploaded_file):
    """
    Processes the uploaded PDF and builds a FAISS vector store.
    """
    if not uploaded_file:
        return None
    
    with st.spinner("Parsing PDF and Creating Vector Store..."):
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Clean up temp file
            os.remove(tmp_path)

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
            doc_splits = text_splitter.split_documents(docs)

            # Embed
            embedding = get_embedding_model()
            vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding)
            
            retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 0.6})
            return retriever
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

def get_llm(provider, model_name, api_key, max_tokens):
    """Factory to create the LLM instance based on provider."""
    if not api_key:
        return None

    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0, max_tokens=max_tokens)
    
    elif provider == "Groq":
        return ChatGroq(model=model_name, api_key=api_key, temperature=0, max_tokens=max_tokens)
    
    elif provider == "OpenRouter":
        # OpenRouter uses the OpenAI client structure with a custom base URL
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=max_tokens
        )
    return None

# --- Core Logic: Adaptive RAG Graph ---

class GraphState(TypedDict):
    """State for the LangGraph."""
    question: str
    generation: str
    web_search: str
    documents: List[Document]

def initialize_graph(llm_model, retriever):
    """
    Builds the Adaptive RAG graph (Retrieval -> Grade -> Transform/Generate -> WebSearch).
    """
    
    # 1. Tools
    web_search_tool = TavilySearchResults(k=3)

    # 2. Components using the provided LLM
    
    # --- Component: Retrieval Grader ---
    class GradeDocuments(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    structured_llm_grader = llm_model.with_structured_output(GradeDocuments)

    system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_grader),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    # --- Component: Question Re-writer ---
    system_rewriter = """You a question re-writer that converts an input question to a better version that is optimized \n 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_rewriter),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.",),
        ]
    )
    question_rewriter = re_write_prompt | llm_model | StrOutputParser()

    # --- Component: Generator (RAG) ---
    # Using standard RAG prompt
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        
        Question: {question} 
        Context: {context} 
        
        Answer:"""
    )
    rag_chain = prompt | llm_model | StrOutputParser()

    # 3. Nodes

    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        print("---CHECK RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        web_search = "No"
        
        for d in documents:
            try:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
                if grade == "yes":
                    filtered_docs.append(d)
                else:
                    continue
            except Exception:
                # Fallback if grading fails
                continue
                
        # If too few relevant docs, trigger web search
        if len(filtered_docs) == 0:
            web_search = "Yes"
            
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def transform_query(state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search_node(state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"] if state["documents"] else []
        
        try:
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            documents.append(web_results)
        except Exception as e:
            st.error(f"Web search failed (check API Key): {e}")
            
        return {"documents": documents, "question": question}

    def decide_to_generate(state):
        print("---DECIDE---")
        web_search = state["web_search"]
        if web_search == "Yes":
            return "transform_query"
        else:
            return "generate"

    # 4. Graph Construction
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# --- Main Application Logic ---

st.title("📄 Corrective RAG with PDF & Web Search")
st.markdown("This assistant uses **Corrective RAG**: It checks if your PDF has the answer. If not, it rewrites your query and searches the web.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Handle Embedding Trigger
if embed_clicked:
    if uploaded_file and api_key:
        st.session_state.retriever = build_vectorstore(uploaded_file)
        if st.session_state.retriever:
            st.success("✅ Document embedded successfully! You can now ask questions.")
    elif not uploaded_file:
        st.error("Please upload a PDF first.")
    elif not api_key:
        st.error("Please enter an API Key first.")

# User Input
user_input = st.chat_input("Ask a question about your document...")

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Run Logic
if user_input:
    # 1. Validation
    if not api_key:
        st.error(f"Please provide your {provider} API Key in the sidebar.")
        st.stop()
    if not st.session_state.retriever:
        st.warning("⚠️ No document embedded yet. I will try to use Web Search only, or please upload/embed a PDF.")
        # Create a dummy retriever if none exists so the graph doesn't crash, 
        # or handle logic to skip retrieval. For now, we allow graph to run, retrieval will return empty, logic triggers web search.
        if not st.session_state.retriever:
             # Basic dummy retriever
             embedding = get_embedding_model()
             empty_vs = FAISS.from_texts([""], embedding)
             st.session_state.retriever = empty_vs.as_retriever()

    # 2. Setup LLM
    llm = get_llm(provider, selected_model, api_key, max_tokens)
    
    # 3. Build Graph (Rebuild on every run to ensure latest LLM config is used)
    app = initialize_graph(llm, st.session_state.retriever)

    # 4. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 5. Run Graph
    inputs = {"question": user_input}
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream events to show reasoning steps
            for event in app.stream(inputs):
                for key, value in event.items():
                    if show_steps:
                        with status_placeholder.expander(f"Agent Step: {key}", expanded=False):
                            st.write(value)
                    
                    if key == "generate":
                        full_response = value.get("generation", "No answer generated.")
            
            message_placeholder.write(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")