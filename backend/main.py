from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
import logging
import shutil
import os
import time
from typing import Optional, Dict, Any
import pytesseract
from functools import lru_cache
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from extractors.image_extractor import ImageExtractor
from extractors.document_extractor import DocumentExtractor
from extractors.audio_extractor import AudioExtractor
from extractors.video_extractor import VideoExtractor
from extractors.excel_extractor import ExcelExtractor
from extractors.email_extractor import EmailExtractor
from extractors.web_extractor import WebExtractor

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure Tesseract path and parameters
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Configure Tesseract parameters for better accuracy
config = '--oem 3 --psm 6 -l eng --dpi 300'

app = FastAPI(title="Text Extraction API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure rate limiting
rate_limit: Dict[str, Dict[str, Any]] = {}
RATE_LIMIT_DURATION = 60  # 1 minute window
MAX_REQUESTS = 30  # Maximum requests per minute
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create upload directory
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Thread pool for concurrent processing
thread_pool = ThreadPoolExecutor(max_workers=4)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean up old rate limit entries
    if client_ip in rate_limit:
        if current_time - rate_limit[client_ip]['start_time'] > RATE_LIMIT_DURATION:
            del rate_limit[client_ip]
    
    # Check rate limit
    if client_ip in rate_limit:
        if rate_limit[client_ip]['count'] >= MAX_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."}
            )
        rate_limit[client_ip]['count'] += 1
    else:
        rate_limit[client_ip] = {
            'count': 1,
            'start_time': current_time
        }
    
    response = await call_next(request)
    return response

# Check if tesseract is installed
def check_tesseract() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.error(f"Tesseract not properly configured: {str(e)}")
        return False

# Check if poppler is installed (required for PDF processing)
def check_poppler() -> bool:
    try:
        # Check if pdftoppm (part of poppler-utils) is available
        result = shutil.which('pdftoppm')
        if result is None:
            logger.error("pdftoppm (poppler-utils) not found in PATH")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking poppler installation: {str(e)}")
        return False

@app.get("/health")
async def health_check():
    checks = {
        "tesseract": check_tesseract(),
        "poppler": check_poppler()
    }
    return {"status": "healthy", "dependencies": checks}

# Initialize extractors
image_extractor = ImageExtractor()
document_extractor = DocumentExtractor()
audio_extractor = AudioExtractor()
video_extractor = VideoExtractor()
excel_extractor = ExcelExtractor()
email_extractor = EmailExtractor()
web_extractor = WebExtractor()

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Validate file size
        file_size = 0
        chunk_size = 8192  # 8KB chunks
        content = bytearray()

        # Reset file position to start
        await file.seek(0)
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content.extend(chunk)
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)}MB"
                )
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file provided")

        # Get file extension
        file_extension = '.' + file.filename.lower().split('.')[-1]
        
        try:
            # Process based on file type
            if file_extension in ['.png', '.jpg', '.jpeg']:
                if not check_tesseract():
                    raise HTTPException(status_code=500, detail="OCR dependencies not configured")
                text = image_extractor.extract_text_from_image(content)
            
            elif file_extension in ['.pdf', '.docx', '.txt']:
                if file_extension == '.pdf' and not check_poppler():
                    raise HTTPException(status_code=500, detail="PDF dependencies not configured")
                text = document_extractor.extract_text(content, file_extension)
            
            elif file_extension in ['.xlsx', '.xls']:
                text = excel_extractor.extract_text_from_excel(content, file_extension)
            
            elif file_extension == '.eml':
                text = email_extractor.extract_text_from_email(content)
            
            elif file_extension == '.html':
                text = web_extractor.extract_text_from_html(content.decode())
            
            elif file_extension in ['.mp3', '.wav']:
                text = audio_extractor.extract_text_from_audio(content, file_extension)
            
            elif file_extension in ['.mp4', '.avi', '.mov']:
                text = video_extractor.extract_text_from_video(content, file_extension)
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

            return {"extracted_text": text}

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            try:
                text = content.decode('utf-8')
            except Exception as e:
                logger.error(f"Error processing TXT: {str(e)}")
                raise HTTPException(status_code=500, detail="Error processing text file")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        if not text.strip():
            return JSONResponse(
                status_code=200,
                content={"extracted_text": "No text was extracted from the file. The file might be empty or contain no readable text."}
            )
        
        return {"extracted_text": text}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    finally:
        await file.close()
