"""Preprocess textual data module."""
import re
from io import BytesIO
from typing import List, Union

import streamlit as st
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader


@st.cache_data
def parse_pdf(pdf_data: BytesIO) -> List[str]:
    """Parses PDF data from buffer.

    Parameters:
    -----------
    pdf_data: BytesIO
        PDF file data.

    Returns:
    --------
    content: List[str]
        PDF pages corpus.
    """
    pdf = PdfReader(pdf_data)
    pages_corpus = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        pages_corpus.append(text)

    return pages_corpus


@st.cache_data
def text_to_docs(text: Union[str, List[str]]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata.

    Parameters:
    -----------
    text: str or List[str]
        Pages text.

    Returns:
    --------
    doc_chunks: List[Document]
        Pages documents metadata.
    """
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)

    return doc_chunks


@st.cache_data
def index_embedding(
    api_key: str,
    pages: List[Document]):
    """
    Index embeddings and store in a Meta vector database (FAISS).

    Parameters:
    -----------
    api_key: str
        OpenAI api key.
    pages: List[Document]
        Pages documents metadata.

    Returns:
    --------
    index: FAISS
        Meta vector store.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✅")
    return index
