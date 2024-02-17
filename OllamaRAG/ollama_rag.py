
# source: https://github.com/MikeyBeez/SimpleAgent/blob/main/ollama/RAGtool3.py

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr


# Initialize variables to store the previous URL and its corresponding data and embeddings
prev_url = None
prev_data = None
prev_vectorstore = None


def process_url_and_question(url: str, question: str):
    global prev_url, prev_data, prev_vectorstore
    if url != prev_url:
        loader = WebBaseLoader(url)
        prev_data = loader.load()
        prev_url = url
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        all_splits = text_splitter.split_documents(prev_data)
        prev_vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    ollama = Ollama(base_url="http://localhost:11434", model="mistral")
    qachain = RetrievalQA.from_chain_type(ollama, retriever=prev_vectorstore.as_retriever())

    result = qachain({"query": question})
    return result


iface = gr.Interface(fn=process_url_and_question,
                     inputs=["text", "text"],
                     outputs="text")
iface.launch()
