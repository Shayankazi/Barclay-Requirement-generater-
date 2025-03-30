import os
from pdf2image import convert_from_bytes
from docx import Document
from .image_extractor import ImageExtractor
import io
import tempfile
import threading
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging
import fitz  # PyMuPDF
import time
from PIL import Image
import numpy as np
import re
from cachetools import LRUCache
from threading import Lock

logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self):
        self.image_extractor = ImageExtractor()
        # Use LRUCache for better performance
        self.cache = LRUCache(maxsize=100)
        self.cache_lock = Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        # PDF extraction settings
        self.min_text_length = 50  # Minimum length to consider text extraction successful
        self.max_dpi = 300  # Maximum DPI to use for OCR
        self.min_dpi = 150  # Minimum DPI to use for OCR
        self.default_dpi = 200  # Default DPI for OCR
        self.ocr_threshold = 0.6  # Confidence threshold for OCR

    def compute_hash(self, file_bytes):
        """Compute hash of file bytes for caching"""
        return hashlib.sha256(file_bytes).hexdigest()

    def extract_text_from_pdf_direct(self, file_bytes):
        """Extract text directly from PDF without OCR"""
        try:
            # Load PDF from memory
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
                total_pages = len(pdf_document)
                logger.info(f"PDF has {total_pages} pages")
                
                # Prepare futures for parallel processing
                futures = []
                for page_num in range(total_pages):
                    futures.append(self.thread_pool.submit(
                        self.process_pdf_page_direct, pdf_document, page_num
                    ))
                
                # Collect results
                page_texts = []
                for future in futures:
                    page_text = future.result()
                    if page_text:
                        page_texts.append(page_text)
                
                # Join all page texts
                full_text = "\n\n".join(page_texts)
                
                # Check if we got enough text
                if len(full_text.strip()) > self.min_text_length:
                    logger.info("Successfully extracted text directly from PDF")
                    return full_text, True
                else:
                    logger.info("Direct text extraction yielded insufficient text, will try OCR")
                    return full_text, False
                
        except Exception as e:
            logger.warning(f"Error in direct PDF text extraction: {str(e)}")
            return "", False

    def process_pdf_page_direct(self, pdf_document, page_num):
        """Process a single PDF page for text extraction"""
        try:
            page = pdf_document[page_num]
            
            # Get text with enhanced options
            text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            
            # If text is too short, try alternative extraction methods
            if len(text.strip()) < 20:
                # Try extracting with different flags
                text = page.get_text("text", sort=True, flags=fitz.TEXT_DEHYPHENATE)
                
                # If still too short, try extracting as blocks
                if len(text.strip()) < 20:
                    blocks = page.get_text("blocks")
                    if blocks:
                        text = "\n".join([b[4] for b in blocks])
            
            # Clean the text
            text = self.clean_pdf_text(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error processing PDF page {page_num}: {str(e)}")
            return ""

    def clean_pdf_text(self, text):
        """Clean text extracted from PDF"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines while preserving paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text.strip()

    def estimate_optimal_dpi(self, pdf_document):
        """Estimate optimal DPI based on document complexity"""
        try:
            # Sample a few pages to determine complexity
            num_pages = len(pdf_document)
            sample_pages = min(3, num_pages)
            
            total_chars = 0
            total_images = 0
            
            for i in range(min(sample_pages, num_pages)):
                page = pdf_document[i]
                
                # Count text characters
                text = page.get_text("text")
                total_chars += len(text)
                
                # Count images
                image_list = page.get_images()
                total_images += len(image_list)
            
            # Calculate average complexity
            avg_chars = total_chars / sample_pages
            avg_images = total_images / sample_pages
            
            # Determine DPI based on complexity
            if avg_chars > 1000 or avg_images > 5:
                # Complex document with lots of text or images
                return self.max_dpi
            elif avg_chars < 200 and avg_images < 2:
                # Simple document with little text and few images
                return self.min_dpi
            else:
                # Medium complexity
                return self.default_dpi
                
        except Exception as e:
            logger.warning(f"Error estimating optimal DPI: {str(e)}")
            return self.default_dpi

    def extract_from_pdf(self, file_bytes):
        """Extract text from PDF using both direct extraction and OCR with optimizations"""
        try:
            start_time = time.time()
            
            # Check cache first
            file_hash = self.compute_hash(file_bytes)
            with self.cache_lock:
                if file_hash in self.cache:
                    logger.info("Using cached PDF result")
                    return self.cache[file_hash]
            
            # Step 1: Try direct text extraction first (much faster)
            direct_text, direct_success = self.extract_text_from_pdf_direct(file_bytes)
            
            # If direct extraction was successful, return the result
            if direct_success:
                # Cache the result
                with self.cache_lock:
                    self.cache[file_hash] = direct_text
                
                logger.info(f"PDF extraction completed in {time.time() - start_time:.2f}s using direct extraction")
                return direct_text
            
            # Step 2: If direct extraction failed, use OCR with optimized parameters
            logger.info("Direct extraction insufficient, falling back to OCR")
            
            # Determine optimal DPI based on document complexity
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
                optimal_dpi = self.estimate_optimal_dpi(pdf_document)
                logger.info(f"Using optimal DPI: {optimal_dpi}")
            
            # Convert PDF to images with optimized parameters
            images = convert_from_bytes(
                file_bytes,
                dpi=optimal_dpi,
                thread_count=8,  # Use more threads for conversion
                use_pdftocairo=True,  # Use pdftocairo which is faster
                grayscale=True,  # Convert to grayscale for better OCR
                size=(None, 2000)  # Limit height to 2000px for speed while maintaining quality
            )
            
            # Extract text from each page in parallel with improved batching
            # Process images in batches to avoid memory issues
            batch_size = 4
            all_text = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                
                # Process batch in parallel
                futures = []
                for image in batch:
                    futures.append(self.thread_pool.submit(
                        self.process_pdf_page_ocr, image, i
                    ))
                
                # Collect batch results
                for future in futures:
                    try:
                        page_text = future.result()
                        if page_text:
                            all_text.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from PDF page: {str(e)}")
            
            # Combine text from all pages
            if direct_text and len(direct_text) > self.min_text_length:
                # If we have some direct text, combine it with OCR results
                result = direct_text + "\n\n" + "\n\n".join(all_text)
            else:
                # Otherwise just use OCR results
                result = "\n\n".join(all_text)
            
            # Post-process the combined text
            result = self.post_process_pdf_text(result)
            
            # Cache the result
            with self.cache_lock:
                self.cache[file_hash] = result
            
            logger.info(f"PDF extraction completed in {time.time() - start_time:.2f}s using OCR")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def process_pdf_page_ocr(self, image, page_num):
        """Process a single PDF page with OCR"""
        try:
            # Enhanced preprocessing for PDF images
            enhanced_image = self.enhance_pdf_image(image)
            
            # Extract text using the image extractor
            text = self.image_extractor.extract_text_from_pdf_image(enhanced_image)
            
            # Add page number for reference
            return f"Page {page_num + 1}:\n{text}"
            
        except Exception as e:
            logger.warning(f"Error processing PDF page {page_num} with OCR: {str(e)}")
            return ""

    def enhance_pdf_image(self, image):
        """Apply PDF-specific enhancements to improve OCR quality"""
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Check if image is already grayscale
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Convert to grayscale
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
                gray = gray.astype(np.uint8)
            else:
                gray = img_array
            
            # Create PIL image from array
            enhanced = Image.fromarray(gray)
            
            # Increase contrast slightly
            enhancer = Image.ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
            
            # Increase sharpness
            enhancer = Image.ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.5)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error enhancing PDF image: {str(e)}")
            return image  # Return original if enhancement fails

    def post_process_pdf_text(self, text):
        """Apply post-processing to improve extracted PDF text"""
        if not text:
            return ""
            
        # Remove duplicate lines that often occur in PDFs
        lines = text.split('\n')
        unique_lines = []
        prev_line = ""
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                if prev_line:  # Only add newline if we have content
                    unique_lines.append("")
                prev_line = ""
                continue
                
            # Skip duplicate lines
            if line.strip() == prev_line.strip():
                continue
                
            unique_lines.append(line)
            prev_line = line
        
        # Join lines back together
        text = '\n'.join(unique_lines)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize newlines
        
        return text.strip()

    def extract_from_docx(self, file_bytes):
        """Extract text from DOCX files"""
        try:
            # Check cache first
            file_hash = self.compute_hash(file_bytes)
            with self.cache_lock:
                if file_hash in self.cache:
                    logger.info("Using cached DOCX result")
                    return self.cache[file_hash]
            
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Open and extract text from document
            doc = Document(tmp_path)
            text = []
            
            # Extract text from paragraphs
            for para in doc.paragraphs:
                if para.text.strip():  # Only add non-empty paragraphs
                    text.append(para.text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():  # Only add non-empty cells
                            row_text.append(cell.text)
                    if row_text:  # Only add non-empty rows
                        text.append(' | '.join(row_text))
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            result = '\n'.join(text)
            
            # Cache the result
            with self.cache_lock:
                self.cache[file_hash] = result
            
            return result
            
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    @lru_cache(maxsize=5)
    def get_encoding_list(self):
        """Return list of encodings to try, cached to avoid recreation"""
        return ['utf-8', 'latin-1', 'ascii', 'utf-16', 'windows-1252']

    def extract_from_txt(self, file_bytes):
        """Extract text from TXT files"""
        try:
            # Check cache first
            file_hash = self.compute_hash(file_bytes)
            with self.cache_lock:
                if file_hash in self.cache:
                    logger.info("Using cached TXT result")
                    return self.cache[file_hash]
            
            # Decode bytes with different encodings
            encodings = self.get_encoding_list()
            text = None
            
            for encoding in encodings:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise Exception("Could not decode text file with supported encodings")
            
            # Cache the result
            with self.cache_lock:
                self.cache[file_hash] = text
            
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")

    def extract_text_from_document(self, file_bytes, file_extension):
        """Main method to extract text based on file type"""
        extractors = {
            '.pdf': self.extract_from_pdf,
            '.docx': self.extract_from_docx,
            '.txt': self.extract_from_txt
        }
        
        if file_extension not in extractors:
            raise ValueError(f"Unsupported document type: {file_extension}")
        
        return extractors[file_extension](file_bytes)