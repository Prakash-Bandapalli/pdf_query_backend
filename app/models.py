from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class AnswerResponse(BaseModel):
    document_id: str
    question: str
    answer: str
