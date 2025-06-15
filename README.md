# 🧠 AI-Powered Arabic Content Analysis Agent

This project is an AI-powered assistant built using the Retrieval-Augmented Generation (RAG) framework to analyze **Arabic content** from PDFs and images. The assistant can answer user queries in **Arabic** via **text** and optionally support **voice** interfaces.

## 🔍 Features

- Upload **Arabic PDF documents** and **images**
- Extract and chunk text for embedding
- **OCR support** for Arabic text in images
- Embedding generation using OpenAI's `text-embedding-3-large`
- In-memory vector store for fast retrieval
- RAG-based question-answering in Arabic using `gpt-4o-mini`
- Gradio-based **text and file upload UI**

## 🛠 Tech Stack

- **LangChain**: For RAG pipeline and document handling
- **OpenAI API**: For embeddings, LLM responses, and OCR
- **Gradio**: For building a simple web interface
- **PyMuPDF**: To load Arabic text from PDFs
- **Base64 + Vision API**: To extract text from images

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 2. Set Environment Variables

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_key_here
```

### 3. Run the App

```bash
python app.py
```

### 3. Run the Agent

```bash
python arabic_rag.py
```

> The Gradio app will open in your browser.

## 📁 Project Structure

```
├── arabic_rag.py         # Main RAG logic and agent flow
├── app.py                # Gradio interface for upload and chat
├── agent_utils.py        # Image OCR, PDF parsing, doc creation
├── .env                  # API keys
└── README.md             # Project documentation
```

## ✅ Deliverables Summary

* ✅ Working prototype with RAG on Arabic content
* ✅ Source code with documentation and clean structure
* ✅ Gradio-based UI for chat and document uploads
* ✅ OCR pipeline for image-based Arabic text

## ⚠️ Notes

* Currently uses **InMemoryVectorStore** – not suitable for large-scale apps.
* Voice interface (optional) can be added using `sounddevice` or Gradio `audio` components.
* Make sure image quality is clear for accurate OCR.

---
