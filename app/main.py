# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import io

# Import components and service functions
from . import services
from . import models
from .config import LLM, EMBEDDING_MODEL, TABLE_NAME # Although not used directly here, ensures init runs

# Initialize FastAPI app
app = FastAPI(title="PDF Question Answering API")

# --- CORS Middleware ---
# Allows requests from your React frontend (adjust origins as needed)
origins = [
    "http://localhost:3000",  # Default React dev server port
    "http://localhost:5173",  # Default Vite dev server port
    "https://pdf-query-frontend.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---

@app.get("/", summary="Root endpoint for health check")
async def read_root():
    """Simple health check endpoint."""
    return {"status": "API is running", "astra_table": TABLE_NAME}

@app.post("/upload", response_model=models.UploadResponse, summary="Upload a PDF for indexing")
async def upload_pdf(file: UploadFile = File(..., description="PDF file to upload")):
    """
    Handles PDF file upload, text extraction, chunking, embedding,
    and indexing into Astra DB.
    Returns a unique document ID for querying.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    try:
        # Read file content into memory as bytes
        pdf_bytes = await file.read()
        if not pdf_bytes:
             raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        print(f"Received file: {file.filename}, size: {len(pdf_bytes)} bytes")

        # Call the indexing service function
        doc_id = services.index_pdf_text(pdf_bytes, file.filename)

        return models.UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            message="PDF processed and indexed successfully."
        )

    except ValueError as ve: # Catch specific errors from text extraction/splitting
         print(f"Value error during upload: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: # Catch errors from indexing/DB operations
         print(f"Runtime error during upload: {re}")
         raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Unexpected error during upload: {e}")
        # Log the full error for debugging
        # logger.exception("Unexpected error during PDF upload") # If using logging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file processing: {e}")
    finally:
        # Ensure file handle is closed (FastAPI usually handles this, but good practice)
        await file.close()


@app.post("/ask", response_model=models.AnswerResponse, summary="Ask a question about an uploaded PDF")
async def ask_question(request: models.QuestionRequest):
    """
    Receives a document ID and a question, retrieves relevant context
    from Astra DB (filtered by doc_id), and uses the LLM to generate an answer.
    """
    if not request.document_id or not request.question:
         raise HTTPException(status_code=400, detail="Both 'document_id' and 'question' are required.")

    try:
        print(f"Received question for doc_id: {request.document_id}")
        answer = services.answer_question(request.document_id, request.question)
        return models.AnswerResponse(
            document_id=request.document_id,
            question=request.question,
            answer=answer
        )
    except ValueError as ve: # Could be raised if doc_id format is invalid (if you add validation)
         print(f"Value error during query: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: # Catch errors from LLM/DB query
         print(f"Runtime error during query: {re}")
         raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Unexpected error during query: {e}")
        # logger.exception("Unexpected error during question answering") # If using logging
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while answering the question: {e}")

