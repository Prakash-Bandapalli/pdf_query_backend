from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import io

# Import components and service functions
from . import services
from . import models
from .config import LLM, EMBEDDING_MODEL, TABLE_NAME # Ensures config.py's initialization runs
from .keep_alive import lifespan_manager # this basically visits the backend server perdiocally to avoid slow downs

# Initialize FastAPI app with the lifespan manager from keep_alive.py
app = FastAPI(title="PDF Question Answering API", lifespan=lifespan_manager)

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://pdf-query-frontend.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.api_route("/", methods=["GET", "HEAD"], summary="Root endpoint for health check and keep-alive")
async def read_root(request: Request):
    if request.method == "HEAD":
        return None
    return {"status": "API is running", "astra_table": TABLE_NAME}

@app.post("/upload", response_model=models.UploadResponse, summary="Upload a PDF for indexing")
async def upload_pdf(file: UploadFile = File(..., description="PDF file to upload")):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
        
        print(f"Received file: {file.filename}, size: {len(pdf_bytes)} bytes")
        doc_id = services.index_pdf_text(pdf_bytes, file.filename)
        
        return models.UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            message="PDF processed and indexed successfully."
        )
    except ValueError as ve:
        print(f"Value error during upload: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        print(f"Runtime error during upload: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file processing: {e}")
    finally:
        await file.close()

@app.post("/ask", response_model=models.AnswerResponse, summary="Ask a question about an uploaded PDF")
async def ask_question(request: models.QuestionRequest):
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
    except ValueError as ve:
        print(f"Value error during query: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        print(f"Runtime error during query: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Unexpected error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while answering the question: {e}")