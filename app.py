import os
import json
import logging
import ngrok
import google.generativeai as genai
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB9_uEfeyLvJ1O-PrT8Qlj8PlOG-p_MvsU")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2lOmx8bvmkfgMfZiC3ROucOdD1P_5RhV34WaJzsyKtTue89x7")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
else:
    logger.error("GOOGLE_API_KEY not set!")

# Load YOLO model
model_path = "best.pt"
yolo_model = YOLO(model_path)

# Load FAISS for RAG
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)  # Adjust dimensions as needed
documents = []

def get_embedding(text: str):
    return np.array(embedding_model.encode(text)).astype('float32')

class ChatRequest(BaseModel):
    message: str

class SearchRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    query_embedding = get_embedding(request.message)
    _, indices = index.search(query_embedding.reshape(1, -1), 1)
    context = documents[indices[0][0]] if documents else ""
    full_query = f"{context}\n\n{request.message}"
    response = model.generate_content(full_query)
    if response and hasattr(response, "candidates"):
        return {"response": response.candidates[0].content}
    raise HTTPException(status_code=500, detail="Failed to generate response")

@app.post("/search")
async def search_market_price(request: SearchRequest):
    # Placeholder for actual implementation
    return {"response": f"Searching for {request.query}"}

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        pdf_content = ""
        with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
            for page in pdf:
                pdf_content += page.get_text("text")
        return {"text": pdf_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
