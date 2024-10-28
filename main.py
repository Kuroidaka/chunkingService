from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import extract_text_from_pdf
# from chunkingStratigy import agentic_chunking
from chunkingStratigy import agenticChunking
import io
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(
    port=9001
)


@app.post("/chunking/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    file_like = io.BytesIO(content)  # Wrap bytes in a BytesIO object
    text_content = extract_text_from_pdf(file_like)  # Pass BytesIO object to the extraction function
    chunks = agenticChunking(text_content)


    return JSONResponse(chunks)
