from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from agent_utils import (
    extract_text_from_image,
    extract_text_from_pdf,
    create_docs_for_image_data
)
from typing_extensions import List, TypedDict
load_dotenv()


template = """You are an AI Assistant speak in Arabic. You have two options to perform that task. Respond for basic queries or the context. In case, the question is related to context and it is not available, tell you don't know. 

You may be asked in English about the context, respond in Arabic

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


files = [    
    # "/home/mohdvasm/Downloads/WhatsApp Image 2025-06-15 at 14.45.04 (1).jpeg", 
    # "/home/mohdvasm/Downloads/WhatsApp Image 2025-06-15 at 14.45.04.jpeg",
    # "/home/mohdvasm/Documents/tipu sultan.pdf",
    # "/home/mohdvasm/Downloads/Demo123.pdf"
]

# One function to load pdf and image both based on file ext
def load_docs(files: list):
    try:
        docs = []
        for i, file_path in enumerate(files):
            print(f"Processing: {file_path}")
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                for page in loader.lazy_load():
                    docs.append(page)

            elif file_path.endswith((".jpg", ".jpeg", ".png")):
                img_content = extract_text_from_image(file_path)
                doc = create_docs_for_image_data(
                    img_content,
                    file_path
                    )
                docs.append(doc)
        return docs
    except Exception as e:
        print(f"Error: {e}")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# For testing agent in terminal
if files:
    docs = load_docs(files)
    
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Relevant doc retriever
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    print(retrieved_docs)
    return {"context": retrieved_docs}

# Response generator
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# Custom function to update vector store based on chat files or independent files from knowledge base tab
def update_vectorstore(files: list):
    try:
        print(f"Updating vector store")
        docs = load_docs(files)
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        _ = vector_store.add_documents(documents=all_splits)
        print(f"Vector store is updated")
    except Exception as e:
        print(f"Error: {error}")


if __name__ == "__main__":
    print("Arabic RAG Application")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            response = graph.invoke({"question": user_input})
            print(f'Assistant: {response["answer"]}')
        except:
            # fallback if input() is not available
            user_input = "Hello"
            print("User: " + user_input)
            response = graph.invoke({"question": user_input})
            break


