# app/services.py
import fitz # PyMuPDF
import uuid
import io
from typing import List, Dict, Tuple
from langchain_community.vectorstores import Cassandra
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document # To structure text chunks with metadata
from langchain.chains import RetrievalQA

# Import initialized components from config
from .config import LLM, EMBEDDING_MODEL, TABLE_NAME

# --- Text Processing Functions ---

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
        doc.close()
        return raw_text
    except Exception as e:
        print(f"Error extracting text from PDF bytes: {e}")
        raise ValueError(f"Failed to process PDF content: {e}")


def split_text(raw_text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Splits text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

# --- Vector Store and Indexing Functions ---

def get_vector_store() -> Cassandra:
    """Gets an instance of the Cassandra vector store."""
    return Cassandra(
        embedding=EMBEDDING_MODEL,
        table_name=TABLE_NAME,
        # session and keyspace are handled by cassio.init()
    )

def index_pdf_text(pdf_bytes: bytes, filename: str) -> str:
    """
    Extracts text, splits, embeds, and stores it in Astra DB
    associated with a unique document ID.
    Returns the generated document ID.
    """
    print(f"Starting indexing for file: {filename}")
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text:
        raise ValueError("No text could be extracted from the PDF.")

    texts = split_text(raw_text)
    if not texts:
        raise ValueError("Text splitting resulted in empty chunks.")

    doc_id = str(uuid.uuid4())
    print(f"Generated document ID: {doc_id}")

    # Create LangChain Document objects with metadata
    documents_with_metadata: List[Document] = []
    for i, text_chunk in enumerate(texts):
        metadata = {
            "doc_id": doc_id,
            "source_filename": filename,
            "chunk_index": i # Optional: Useful for debugging or citing sources
        }
        documents_with_metadata.append(Document(page_content=text_chunk, metadata=metadata))

    # Get vector store instance
    vector_store = get_vector_store()

    # Add texts with metadata to Astra DB
    try:
        print(f"Adding {len(documents_with_metadata)} chunks to Astra DB for doc_id {doc_id}...")
        inserted_ids = vector_store.add_documents(documents_with_metadata, batch_size=20) # Adjust batch_size if needed
        print(f"Successfully inserted {len(inserted_ids)} vectors into table '{TABLE_NAME}'.")
        return doc_id
    except Exception as e:
        print(f"Error adding documents to Astra DB: {e}")
        raise RuntimeError(f"Failed to index document in Astra DB: {e}")


# --- Question Answering Functions ---

def answer_question(doc_id: str, question: str) -> str:
    """
    Answers a question based on the indexed content of a specific document ID.
    """
    print(f"Answering question for doc_id: {doc_id}")
    vector_store = get_vector_store()

    # --- CRUCIAL: Filtering by doc_id ---
    # Create a retriever that filters by metadata BEFORE sending chunks to the LLM
    retriever = vector_store.as_retriever(
        search_kwargs={
            'k': 3, # Number of relevant chunks to retrieve
            # This filter syntax depends on the LangChain Cassandra integration.
            # Verify the exact filter format if this doesn't work.
            # Common patterns involve `filter` or directly in search_kwargs.
            # For CassIO/AstraDB, metadata filtering is often direct key-value:
             "filter": {"doc_id": doc_id} # Adjust this if syntax differs in your version
        }
    )

    # --- Check if any relevant chunks were found ---
    # Perform a preliminary search to see if chunks exist for this doc_id
    # Note: This adds an extra query but prevents sending empty context to LLM
    try:
        relevant_docs = retriever.get_relevant_documents(question)
        if not relevant_docs:
            print(f"No relevant text chunks found for doc_id '{doc_id}' matching the question.")
            return "I could not find any relevant information in the specified document to answer your question."
        print(f"Retrieved {len(relevant_docs)} relevant chunks for doc_id '{doc_id}'.")
    except Exception as e:
        # This might happen if the filter syntax is wrong or DB issue
        print(f"Error retrieving documents for doc_id '{doc_id}': {e}")
        return "An error occurred while trying to retrieve information for the specified document."


    # --- Use RetrievalQA Chain ---
    # This chain combines retrieval and LLM question answering
    qa_chain = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff", # "stuff" puts all retrieved chunks into the prompt context
        retriever=retriever,
        return_source_documents=False # Set to True if you want source snippets
    )

    try:
        print("Sending query to LLM...")
        result = qa_chain({"query": question}) # Use invoke for newer LangChain versions: qa_chain.invoke(question)
        print("Received answer from LLM.")
        return result.get("result", "Sorry, I could not generate an answer.").strip()
    except Exception as e:
        print(f"Error during LLM query: {e}")
        raise RuntimeError(f"Failed to get answer from LLM: {e}")