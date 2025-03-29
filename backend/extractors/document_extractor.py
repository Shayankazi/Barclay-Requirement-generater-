import os
from pdf2image import convert_from_bytes
from docx import Document
from .image_extractor import ImageExtractor
import io
import tempfile

class DocumentExtractor:
    def __init__(self):
        self.image_extractor = ImageExtractor()

    def extract_from_pdf(self, file_bytes):
        """Extract text from PDF using both OCR and direct extraction"""
        try:
            # First, convert PDF to images
            images = convert_from_bytes(file_bytes)
            
            # Extract text from each page
            extracted_text = []
            for image in images:
                # Extract text using OCR
                page_text = self.image_extractor.extract_text_from_pdf_image(image)
                extracted_text.append(page_text)
            
            # Combine text from all pages
            return '\n\n'.join(extracted_text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_from_docx(self, file_bytes):
        """Extract text from DOCX files"""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Open and extract text from document
            doc = Document(tmp_path)
            text = []
            
            # Extract text from paragraphs
            for para in doc.paragraphs:
                text.append(para.text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text)
                    text.append(' | '.join(row_text))
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return '\n'.join(text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def extract_from_txt(self, file_bytes):
        """Extract text from TXT files"""
        try:
            # Decode bytes with different encodings
            encodings = ['utf-8', 'latin-1', 'ascii']
            text = None
            
            for encoding in encodings:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise Exception("Could not decode text file with supported encodings")
            
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")

    def extract_text(self, file_bytes, file_extension):
        """Main method to extract text based on file type"""
        extractors = {
            '.pdf': self.extract_from_pdf,
            '.docx': self.extract_from_docx,
            '.txt': self.extract_from_txt
        }
        
        if file_extension not in extractors:
            raise ValueError(f"Unsupported document type: {file_extension}")
        
        return extractors[file_extension](file_bytes)
