# app/config.py
import os
import cassio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_TABLE_NAME = os.getenv("ASTRA_DB_TABLE_NAME", "pdf_qa_documents_gemini") # Default if not in .env

def initialize_components():
    """Initializes and returns LLM, Embeddings model, and Vector Store Table Name."""
    if not GOOGLE_API_KEY or not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID:
        raise ValueError("Missing necessary environment variables for Google API or Astra DB.")

    try:
        # Initialize CassIO connection (singleton pattern, safe to call multiple times)
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

        # Initialize Google LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1, # Lower temperature for more factual answers
            convert_system_message_to_human=True
        )

        # Initialize Google Embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", # Or your preferred Google embedding model
            google_api_key=GOOGLE_API_KEY
        )

        print(f"Components initialized. Using Astra DB table: {ASTRA_DB_TABLE_NAME}")
        return llm, embedding_model, ASTRA_DB_TABLE_NAME

    except Exception as e:
        print(f"Error during component initialization: {e}")
        raise RuntimeError(f"Failed to initialize components: {e}")

# Initialize components on module load
try:
    LLM, EMBEDDING_MODEL, TABLE_NAME = initialize_components()
except Exception as e:
    # Handle initialization failure gracefully, maybe exit or log critical error
    print(f"CRITICAL: Failed to initialize application components. Exiting. Error: {e}")
    exit(1) # Or raise an exception that the main app can catch