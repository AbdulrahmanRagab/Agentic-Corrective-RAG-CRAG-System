# 📄 Corrective RAG with PDF & Web Search

A Streamlit application implementing **Corrective RAG (Retrieval-Augmented Generation)**: the LLM retrieves context from your uploaded PDF, grades document relevance, rewrites the query if retrieval fails, falls back to web search when needed, and generates answers grounded in the retrieved sources.

---

## 🧭 Table of Contents

*   [🤖 Overview](#-overview)
*   [✨ Features](#-features)
*   [🧰 Tech Stack](#-tech-stack)
*   [🧠 Architecture](#-architecture)
*   [🔁 LangGraph Flow](#-langgraph-flow)
*   [📚 Knowledge Base & Retrieval Tools](#-knowledge-base--retrieval-tools)
*   [🚀 Getting Started](#-getting-started)
    *   [✅ Prerequisites](#-prerequisites)
    *   [📦 Installation](#-installation)
    *   [▶️ Run the App](#️-run-the-app)
*   [⚙️ Configuration (UI)](#️-configuration-ui)
*   [🔎 How It Works (Step-by-Step)](#-how-it-works-step-by-step)
*   [🛠️ Customization](#️-customization)
*   [🧯 Troubleshooting](#-troubleshooting)
*   [⚠️ Known Limitations](#️-known-limitations)
*   [🔐 Security Notes](#-security-notes)
*   [🗺️ Roadmap Ideas](#️-roadmap-ideas)
*   [🙏 Acknowledgements / Sources](#-acknowledgements--sources)
*   [📄 License](#-license)
*   [📁 Project Structure](#-project-structure)

---

## 🤖 Overview

This project demonstrates an advanced RAG workflow that goes beyond simple retrieval. Instead of blindly trusting retrieved documents, the system employs a **corrective mechanism**: it grades each retrieved chunk for relevance, and if the local knowledge base lacks sufficient information, it automatically rewrites the query and performs a web search to supplement the answer.

The application provides a user-friendly Streamlit interface where users can upload PDF documents, configure their preferred LLM provider, and interact with an intelligent assistant that transparently shows its reasoning steps.

---

## ✨ Features

*   **Multi-Provider LLM Support**: Seamlessly switch between OpenAI, Groq, and OpenRouter with a variety of models
*   **PDF Document Processing**: Upload and embed PDF documents into a local FAISS vector store
*   **Intelligent Document Grading**: Each retrieved chunk is evaluated for relevance before being used
*   **Automatic Query Rewriting**: When retrieval fails, the system reformulates the question for better results
*   **Web Search Fallback**: Integrates Tavily Search API to fetch real-time information when local knowledge is insufficient
*   **Transparent Reasoning**: Optional display of agent steps showing the workflow execution
*   **MMR Retrieval**: Uses Maximal Marginal Relevance for diverse, relevant document retrieval
*   **Configurable Parameters**: Adjust max tokens, toggle step visibility, and select models from the UI
*   **Session Management**: Clear chat history and reset the knowledge base with one click

---

## 🧰 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Orchestration** | LangGraph |
| **LLM Framework** | LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | HuggingFace (EmbeddingGemma-300M) |
| **PDF Parsing** | PyPDFLoader |
| **Web Search** | Tavily Search API |
| **LLM Providers** | OpenAI, Groq, OpenRouter |

---

## 🧠 Architecture

The application follows a stateful graph-based architecture powered by LangGraph. The core components include:

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Config Panel│  │ PDF Upload  │  │    Chat Interface       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                           │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐   │
│  │ Retrieve │→ │   Grade   │→ │ Transform │→ │  Web Search  │   │
│  └──────────┘  └───────────┘  └───────────┘  └──────────────┘   │
│                      │                              │            │
│                      ▼                              ▼            │
│               ┌─────────────────────────────────────────┐       │
│               │              Generate                   │       │
│               └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  FAISS Vector   │  │    HuggingFace  │  │   Tavily API    │  │
│  │     Store       │  │    Embeddings   │  │   (Web Search)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔁 LangGraph Flow

The corrective RAG workflow follows this decision graph:

```text
                ┌─────────┐
                │  START  │
                └────┬────┘
                     │
                     ▼
              ┌──────────────┐
              │   Retrieve   │
              │  (from PDF)  │
              └──────┬───────┘
                     │
                     ▼
           ┌─────────────────────┐
           │   Grade Documents   │
           │ (relevance check)   │
           └──────────┬──────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
Relevant docs found?        No relevant docs
          │                       │
          ▼                       ▼
   ┌────────────┐        ┌─────────────────┐
   │  Generate  │        │ Transform Query │
   │   Answer   │        │   (rewrite)     │
   └─────┬──────┘        └────────┬────────┘
         │                        │
         │                        ▼
         │               ┌─────────────────┐
         │               │   Web Search    │
         │               │    (Tavily)     │
         │               └────────┬────────┘
         │                        │
         │                        ▼
         │               ┌─────────────────┐
         │               │    Generate     │
         │               │     Answer      │
         │               └────────┬────────┘
         │                        │
         └────────┬───────────────┘
                  │
                  ▼
             ┌─────────┐
             │   END   │
             └─────────┘
```

**Node Descriptions:**

| Node | Purpose |
| :--- | :--- |
| **Retrieve** | Fetches top-k documents from FAISS using MMR |
| **Grade Documents** | Uses structured LLM output to score each document's relevance |
| **Transform Query** | Rewrites the question for better web search results |
| **Web Search** | Queries Tavily API for external information |
| **Generate** | Produces the final answer using retrieved context |

---

## 📚 Knowledge Base & Retrieval Tools

**Local Knowledge Base (FAISS)**

The application uses FAISS (Facebook AI Similarity Search) for efficient vector similarity search. Documents are processed through the following pipeline: PDF files are loaded using `PyPDFLoader`, then text is split into chunks of 600 characters with 150 character overlap using `RecursiveCharacterTextSplitter`. These chunks are embedded using HuggingFace's `EmbeddingGemma-300M` model and stored in the FAISS index. Retrieval uses MMR (Maximal Marginal Relevance) with `k=5` and `lambda_mult=0.6` to balance relevance and diversity.

**Web Search Tool (Tavily)**

When local retrieval fails, the system falls back to Tavily Search API, which is optimized for LLM applications and returns structured, relevant web content. The search returns the top 3 results which are concatenated and added to the document context.

---

## 🚀 Getting Started

### ✅ Prerequisites

Before running this application, ensure you have Python 3.9 or higher installed. You'll also need API keys for at least one LLM provider (OpenAI, Groq, or OpenRouter) and optionally a Tavily API key for web search functionality. The application requires sufficient disk space for the HuggingFace embedding model cache.

### 📦 Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/corrective-rag-assistant.git
cd corrective-rag-assistant
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:

```text
streamlit
langchain
langchain-openai
langchain-groq
langchain-community
langchain-huggingface
langgraph
faiss-cpu
pypdf
pydantic
tavily-python
```

### ▶️ Run the App

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## ⚙️ Configuration (UI)

The sidebar provides all configuration options:

*   **AI Provider Selection**: Choose between OpenAI, Groq, or OpenRouter. Each provider offers different models with varying capabilities and pricing.
*   **API Key Input**: Enter your API key for the selected provider. Keys are handled securely and not stored persistently.
*   **Model Selection**: Available models update based on your provider selection.
*   **Tavily API Key**: Required for web search fallback functionality. Without this key, the system will skip web search when local retrieval fails.
*   **Max Response Tokens**: Control the maximum length of generated responses (50-4096 tokens).
*   **Show Agent Steps**: Toggle to display or hide the reasoning steps during query processing.
*   **Knowledge Base Upload**: Upload PDF documents to build your local knowledge base. Click "Embed Document" after uploading to process the file.
*   **Clear Chat / Reset**: Removes chat history and clears the embedded document.

---

## 🔎 How It Works (Step-by-Step)

The workflow begins when a user submits a question through the chat interface. Here's what happens behind the scenes:

1.  **Retrieval**: The system queries the FAISS vector store using MMR retrieval, fetching the 5 most relevant (and diverse) document chunks based on semantic similarity to the question.
2.  **Document Grading**: Each retrieved document is evaluated by the LLM using structured output. The grader checks if the document contains keywords or semantic meaning related to the question and assigns a binary "yes" or "no" relevance score.
3.  **Decision Point**: If at least one document passes the relevance check, the workflow proceeds directly to generation. If no documents are deemed relevant, the system triggers the corrective path.
4.  **Query Transformation (Corrective Path)**: The LLM rewrites the original question to optimize it for web search, focusing on the underlying semantic intent.
5.  **Web Search (Corrective Path)**: The rewritten query is sent to Tavily Search API, which returns the top 3 relevant web results. These results are converted to document format and added to the context.
6.  **Generation**: The LLM generates a final answer using all available context (filtered local documents and/or web search results). The prompt instructs the model to acknowledge if it doesn't know the answer rather than hallucinating.
7.  **Response Display**: The answer is displayed in the chat interface. If "Show Agent Steps" is enabled, users can expand each step to see the intermediate state values.

---

## 🛠️ Customization

*   **Changing the Embedding Model**: The default embedding model path points to a local cache. Modify the `get_embedding_model()` function to use a different model, e.g.:
    ```python
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ```
*   **Adding New LLM Providers**: Extend the `get_llm()` function to support additional providers by adding new conditional branches with the appropriate client configuration.
*   **Customizing Prompts**: The system prompts for grading, rewriting, and generation are defined in `initialize_graph()`. Modify these to change the LLM's behavior.

---

## 🧯 Troubleshooting

*   **"Error processing PDF"**: Ensure the uploaded file is a valid PDF. Some PDFs with complex formatting or scanned images may not parse correctly. Try a different PDF or use OCR preprocessing.
*   **"Web search failed"**: Verify your Tavily API key is correct and has remaining quota. Check your internet connection and ensure the Tavily service is accessible.
*   **Slow embedding**: The first run downloads the embedding model (~1.2GB). Subsequent runs use the cached model. Consider using a smaller model like MiniLM for faster processing.
*   **Out of memory errors**: Large PDFs with many pages can exhaust memory. Try reducing chunk size or processing smaller documents.
*   **Empty or irrelevant responses**: Ensure your question is related to the uploaded document content. Check if the grading threshold is too strict by examining agent steps.
*   **API rate limits**: When using free tiers, you may hit rate limits. Add delays between requests or upgrade to a paid plan.

---

## ⚠️ Known Limitations

*   Only PDF files are supported; other document formats like DOCX, TXT, or HTML are not processed.
*   The embedding model path is hardcoded to a Windows-specific local path, which requires modification for other operating systems.
*   There is no persistent storage, meaning the vector store is rebuilt on every session and chat history is lost on page refresh.
*   The system processes only one PDF at a time; multiple document support would require modifications.
*   Web search results are concatenated as plain text, potentially losing source attribution.
*   The grading mechanism uses binary classification, which may be too coarse for nuanced relevance assessment.

---

## 🔐 Security Notes

*   API keys entered in the sidebar are stored only in session state and are not persisted to disk or logged. However, keys are transmitted to the respective API providers over HTTPS. For production deployments, consider using environment variables or a secrets manager instead of UI input.
*   Uploaded PDFs are temporarily written to disk during processing and deleted immediately after. Be cautious when uploading sensitive documents.
*   The application does not implement authentication; anyone with access to the URL can use the interface. Consider adding authentication for shared deployments.

---

## 🗺️ Roadmap Ideas

Future enhancements could include:

*   Support for multiple document formats (DOCX, TXT, HTML, Markdown)
*   Persistent vector store with session management
*   Streaming responses for better user experience
*   Source citation with page numbers and confidence scores
*   Multi-document support with document selection
*   Conversation memory for follow-up questions
*   Export functionality for chat history
*   Custom embedding model selection from UI
*   Hallucination detection and fact-checking
*   Deployment configurations for cloud platforms

---

## 🙏 Acknowledgements / Sources

This project builds upon the work of several open-source projects and research:

*   LangChain and LangGraph provide the foundational framework for LLM orchestration and workflow management.
*   The Corrective RAG pattern is inspired by research on self-reflective retrieval systems.
*   FAISS from Meta AI enables efficient similarity search.
*   HuggingFace provides the embedding models and transformers library.
*   Streamlit offers the rapid prototyping framework for the UI.
*   Tavily provides the LLM-optimized search API.

Special thanks to the LangChain community for documentation and examples that informed this implementation.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```text
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 📁 Project Structure

```text
corrective-rag-assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This documentation file
├── LICENSE                 # MIT License
│
├── .gitignore              # Git ignore rules
│   ├── __pycache__/
│   ├── *.pyc
│   ├── .env
│   └── venv/
│
└── assets/                 # (Optional) Images for documentation
    ├── architecture.png
    └── demo.gif
```
