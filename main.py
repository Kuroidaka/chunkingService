from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import extract_text_from_pdf
from typing import List
from pydantic import BaseModel
# from chunkingStratigy import agentic_chunking
from chunkingStratigy import agenticChunking
import io
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(
    port=9001
)

class ParagraphsRequest(BaseModel):
    paragraphs: List[str]

@app.post("/chunking/")
async def upload_file(request: ParagraphsRequest):
    paragraphs = request.paragraphs
    chunks = await agenticChunking(paragraphs)  # Process the paragraphs into chunks
    return JSONResponse(content=chunks)
