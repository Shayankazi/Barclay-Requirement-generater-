from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
import logging
import shutil
import os
import time
import hashlib
from typing import Optional, Dict, Any
import pytesseract
from functools import lru_cache
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvloop
from pydantic import BaseModel

from nlp import RequirementsGenerator

from extractors.image_extractor import ImageExtractor
from extractors.document_extractor import DocumentExtractor
from extractors.audio_extractor import AudioExtractor
from extractors.video_extractor import VideoExtractor
from extractors.excel_extractor import ExcelExtractor
from extractors.email_extractor import EmailExtractor
from extractors.web_extractor import WebExtractor

# Use uvloop for improved async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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
thread_pool = ThreadPoolExecutor(max_workers=8)  # Increased from 4 to 8 for better concurrency

# Result cache
RESULT_CACHE_SIZE = 100
result_cache = {}

# Initialize the RequirementsGenerator
requirements_generator = RequirementsGenerator(
    api_key=os.environ.get("GEMINI_API_KEY"),
    cache_dir=".cache"
)

# Define Pydantic models for request validation
class TextInput(BaseModel):
    text: str
    format_type: str = "standard"  # default to standard format

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean up old rate limit entries
    expired_ips = [ip for ip, data in rate_limit.items() 
                  if current_time - data['start_time'] > RATE_LIMIT_DURATION]
    for ip in expired_ips:
        del rate_limit[ip]
    
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
@lru_cache(maxsize=1)
def check_tesseract() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.error(f"Tesseract not properly configured: {str(e)}")
        return False

@app.get("/health")
async def health_check():
    checks = {
        "tesseract": check_tesseract(),
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

def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content for caching"""
    return hashlib.sha256(content).hexdigest()

async def process_file_content(content: bytes, file_extension: str, filename: str):
    """Process file content based on file type"""
    try:
        # Check cache first
        file_hash = compute_file_hash(content)
        if file_hash in result_cache:
            logger.info(f"Cache hit for file: {filename}")
            return result_cache[file_hash]
        
        # Process based on file type
        if file_extension in ['.png', '.jpg', '.jpeg']:
            if not check_tesseract():
                raise HTTPException(status_code=500, detail="OCR dependencies not configured")
            text = await run_in_threadpool(image_extractor.extract_text_from_image, content)
        
        elif file_extension in ['.pdf', '.docx', '.txt']:
            text = await run_in_threadpool(document_extractor.extract_text_from_document, content, file_extension)
        
        elif file_extension in ['.xlsx', '.xls']:
            text = await run_in_threadpool(excel_extractor.extract_text_from_excel, content, file_extension)
        
        elif file_extension == '.eml':
            text = await run_in_threadpool(email_extractor.extract_text_from_email, content)
        
        elif file_extension == '.html':
            text = await run_in_threadpool(web_extractor.extract_text_from_html, content.decode())
        
        elif file_extension in ['.mp3', '.wav']:
            text = await run_in_threadpool(audio_extractor.extract_text_from_audio, content, file_extension)
        
        elif file_extension in ['.mp4', '.avi', '.mov']:
            text = await run_in_threadpool(video_extractor.extract_text_from_video, content, file_extension)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        # Cache result (maintain cache size)
        if len(result_cache) >= RESULT_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(result_cache))
            del result_cache[oldest_key]
        result_cache[file_hash] = text
        
        return text
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        # Validate file size
        file_size = 0
        chunk_size = 32768  # Increased from 8KB to 32KB for faster reading
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
        
        # Process file content
        text = await process_file_content(bytes(content), file_extension, file.filename)

        if not text.strip():
            return JSONResponse(
                status_code=200,
                content={"extracted_text": "No text was extracted from the file. The file might be empty or contain no readable text."}
            )
        
        # Clean up memory
        del content
        
        # Schedule cleanup of any temporary files
        if background_tasks:
            background_tasks.add_task(lambda: None)  # Placeholder for any cleanup tasks
        
        return {"extracted_text": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    finally:
        await file.close()

# Add a new endpoint for generating requirements from extracted text
@app.post("/generate-requirements")
async def generate_requirements(input_data: TextInput):
    """Generate software requirements from input text."""
    try:
        if not input_data.text or len(input_data.text.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Input text is too short. Please provide more detailed text."
            )
        
        # Process the text using the RequirementsGenerator
        requirements = await run_in_threadpool(
            requirements_generator.generate_requirements_statement,
            input_data.text,
            input_data.format_type
        )
        
        return {
            "requirements": requirements,
            "format": input_data.format_type,
            "input_length": len(input_data.text),
            "output_length": len(requirements)
        }
    
    except Exception as e:
        logger.error(f"Error generating requirements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating requirements: {str(e)}")

# Add a combined endpoint that extracts text from a file and generates requirements
@app.post("/extract-and-generate")
async def extract_and_generate(
    file: UploadFile = File(...),
    format_type: str = "standard",
    background_tasks: BackgroundTasks = None
):
    """Extract text from a file and generate requirements in one step."""
    try:
        # First extract text from the file
        extracted_result = await extract_text(file, background_tasks)
        extracted_text = extracted_result.get("extracted_text", "")
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return JSONResponse(
                status_code=200,
                content={
                    "warning": "The extracted text was too short to generate meaningful requirements.",
                    "extracted_text": extracted_text,
                    "requirements": None
                }
            )
        
        # Generate requirements from the extracted text
        requirements = await run_in_threadpool(
            requirements_generator.generate_requirements_statement,
            extracted_text,
            format_type
        )
        
        return {
            "extracted_text": extracted_text,
            "requirements": requirements,
            "format": format_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract-and-generate: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
