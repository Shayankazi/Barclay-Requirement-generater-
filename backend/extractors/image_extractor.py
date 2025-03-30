import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from typing import List, Tuple, Dict
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self):
        # Initialize EasyOCR with GPU and use it as primary OCR
        self.reader = easyocr.Reader(['en'], gpu=True, download_enabled=False)
        self.initialize_tesseract()
        self.min_confidence = 0.6
        self.cache = {}  # Cache for preprocessed images

    def initialize_tesseract(self):
        """Configure Tesseract parameters for better accuracy"""
        # Use LSTM OCR Engine Mode with automatic page segmentation
        # -l eng: Use English language
        # --oem 1: LSTM OCR Engine
        # --psm 6: Assume a uniform block of text
        # -c tessedit_char_whitelist: Limit to these characters
        self.tesseract_config = r'--oem 1 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()[]{}<>-_=+;:\\/"\' '

    def preprocess_image(self, image) -> List[np.ndarray]:
        """Optimized image preprocessing focusing on most effective techniques"""
        processed_images = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Original grayscale
        processed_images.append(gray)

        # 2. Optimized contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(enhanced)

        # 3. Optimized thresholding
        # Otsu's thresholding with light denoising
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu)

        # 4. Adaptive thresholding (only one optimal size)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(thresh)

        # 5. Morphological operations (only the most effective ones)
        kernel = np.ones((2,2), np.uint8)
        
        # Opening to remove noise
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        processed_images.append(opened)
        
        # Closing to connect components
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(closed)

        return processed_images

    def extract_text_easyocr(self, image) -> Tuple[str, float]:
        """Extract text using EasyOCR with confidence filtering and layout analysis"""
        try:
            results = self.reader.readtext(image)
            if not results:
                return "", 0.0

            # Filter and format results
            text_blocks = []
            total_conf = 0.0
            valid_blocks = 0

            for bbox, text, conf in results:
                if conf > self.min_confidence:
                    # Get text position and layout information
                    x_min = min(point[0] for point in bbox)
                    y_min = min(point[1] for point in bbox)
                    width = max(point[0] for point in bbox) - x_min
                    height = max(point[1] for point in bbox) - y_min

                    # Store position, size, and text
                    text_blocks.append({
                        'x': x_min,
                        'y': y_min,
                        'width': width,
                        'height': height,
                        'text': text,
                        'conf': conf
                    })
                    total_conf += conf
                    valid_blocks += 1

            if not text_blocks:
                return "", 0.0

            # Sort blocks by position (top to bottom, left to right)
            text_blocks.sort(key=lambda x: (x['y'], x['x']))

            # Group blocks into lines based on y-position similarity
            line_threshold = np.mean([block['height'] for block in text_blocks]) * 0.5
            current_line = []
            lines = []

            for block in text_blocks:
                if not current_line or abs(block['y'] - current_line[0]['y']) <= line_threshold:
                    current_line.append(block)
                else:
                    # Sort blocks in current line by x-position
                    current_line.sort(key=lambda x: x['x'])
                    lines.append(current_line)
                    current_line = [block]

            if current_line:
                current_line.sort(key=lambda x: x['x'])
                lines.append(current_line)

            # Build final text with proper spacing
            text_lines = []
            for line in lines:
                line_text = ' '.join(block['text'] for block in line)
                text_lines.append(line_text)

            final_text = '\n'.join(text_lines)
            avg_confidence = total_conf / valid_blocks if valid_blocks > 0 else 0.0

            return final_text, avg_confidence

        except Exception as e:
            logger.error(f"EasyOCR error: {str(e)}")
            return "", 0.0

    def extract_text_tesseract(self, image) -> Tuple[str, float]:
        """Extract text using Tesseract with advanced configuration and layout analysis"""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=self.tesseract_config)
            
            # Filter and format results
            text_blocks = []
            total_conf = 0.0
            valid_blocks = 0
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                conf = float(data['conf'][i])
                if conf > 0:  # Skip negative confidence values
                    text = data['text'][i].strip()
                    if text:
                        # Get position and layout information
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        block_num = data['block_num'][i]
                        line_num = data['line_num'][i]
                        
                        if conf > 60:  # Only include high confidence text
                            text_blocks.append({
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'block': block_num,
                                'line': line_num,
                                'text': text,
                                'conf': conf
                            })
                            total_conf += conf
                            valid_blocks += 1
            
            if not text_blocks:
                return "", 0.0
            
            # Sort blocks by position (top to bottom, then left to right within same block)
            text_blocks.sort(key=lambda x: (x['block'], x['line'], x['x']))
            
            # Group by blocks and lines
            current_block = text_blocks[0]['block']
            current_line = text_blocks[0]['line']
            line_text = []
            block_text = []
            final_text = []
            
            for block in text_blocks:
                if block['block'] != current_block:
                    if line_text:
                        block_text.append(' '.join(line_text))
                    if block_text:
                        final_text.append('\n'.join(block_text))
                    current_block = block['block']
                    current_line = block['line']
                    line_text = [block['text']]
                    block_text = []
                elif block['line'] != current_line:
                    if line_text:
                        block_text.append(' '.join(line_text))
                    current_line = block['line']
                    line_text = [block['text']]
                else:
                    line_text.append(block['text'])
            
            # Add remaining text
            if line_text:
                block_text.append(' '.join(line_text))
            if block_text:
                final_text.append('\n'.join(block_text))
            
            final_str = '\n\n'.join(final_text)
            avg_confidence = total_conf / valid_blocks if valid_blocks > 0 else 0.0
            
            return final_str, avg_confidence
            
        except Exception as e:
            logger.error(f"Tesseract error: {str(e)}")
            return "", 0.0

    def clean_text(self, text: str) -> str:
        """Clean and format extracted text with advanced text refinement"""
        if not text:
            return ""

        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines while preserving paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        # Fix common OCR mistakes
        replacements = {
            'l': 'I',     # Common confusion between l and I
            '0': 'O',     # Common confusion between 0 and O
            '1': 'I',     # Common confusion between 1 and I
            'S': '5',     # Common confusion between S and 5
            'Z': '2',     # Common confusion between Z and 2
            'G': '6',     # Common confusion between G and 6
            'B': '8',     # Common confusion between B and 8
            'rn': 'm',    # Common confusion between 'rn' and 'm'
            'vv': 'w',    # Common confusion between 'vv' and 'w'
        }
        
        # Split into lines to preserve formatting
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            words = line.split()
            cleaned_words = []
            
            for word in words:
                # Fix common OCR mistakes
                if len(word) == 1 and word in replacements:
                    word = replacements[word]
                else:
                    # Fix multi-character confusions
                    for old, new in replacements.items():
                        if len(old) > 1:  # Only apply multi-char replacements
                            word = word.replace(old, new)
                
                # Fix case for common patterns
                if word.isupper() and len(word) > 3:
                    # Convert to title case if it looks like a regular word
                    if not any(c.isdigit() for c in word):
                        word = word.title()
                
                cleaned_words.append(word)
            
            # Join words and add to cleaned lines
            cleaned_lines.append(' '.join(cleaned_words))
        
        # Join lines with proper spacing
        return '\n'.join(cleaned_lines)

    def extract_text_from_image(self, image_bytes):
        """Optimized text extraction focusing on speed and accuracy"""
        try:
            # Convert bytearray to bytes for hashing
            if isinstance(image_bytes, bytearray):
                image_bytes = bytes(image_bytes)
                
            # Check cache first
            cache_key = hash(image_bytes)
            if cache_key in self.cache:
                logger.info("Using cached result")
                return self.cache[cache_key]

            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception("Failed to decode image")
            
            # Get optimized preprocessed versions
            processed_images = self.preprocess_image(image)
            
            # Try EasyOCR first (primary OCR)
            best_text = ""
            best_conf = 0
            
            # Try EasyOCR on original and enhanced images only
            for img in processed_images[:2]:  # Original and CLAHE enhanced
                try:
                    text, conf = self.extract_text_easyocr(img)
                    if text and conf > best_conf:
                        best_text = text
                        best_conf = conf
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {str(e)}")
            
            # If EasyOCR gives good results, use it
            if best_conf > 0.7:
                logger.info(f"Using EasyOCR result with confidence: {best_conf:.2f}")
                final_text = self.clean_text(best_text)
                self.cache[cache_key] = final_text
                return final_text
            
            # If EasyOCR results are poor, try Tesseract on preprocessed images
            tesseract_results = []
            
            for img in processed_images:
                try:
                    text, conf = self.extract_text_tesseract(img)
                    if text and conf > 0:
                        tesseract_results.append({
                            'text': text,
                            'conf': conf
                        })
                except Exception as e:
                    continue  # Skip failed attempts
            
            if tesseract_results:
                # Use the highest confidence Tesseract result
                best_tesseract = max(tesseract_results, key=lambda x: x['conf'])
                if best_tesseract['conf'] > best_conf:
                    best_text = best_tesseract['text']
                    best_conf = best_tesseract['conf']
            
            if not best_text:
                raise Exception("No text could be extracted from the image")
            
            final_text = self.clean_text(best_text)
            
            # Cache the result
            self.cache[cache_key] = final_text
            
            # Log performance metrics
            logger.info(f"Text extracted with confidence: {best_conf:.2f}")
            
            return final_text
            
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")

    def extract_text_from_pdf_image(self, pdf_image):
        """Extract text from PDF-converted image"""
        try:
            # Convert PDF image to bytes
            img_byte_arr = io.BytesIO()
            pdf_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Use the same extraction method
            return self.extract_text_from_image(img_byte_arr)
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF image: {str(e)}")
