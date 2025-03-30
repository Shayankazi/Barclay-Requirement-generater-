import pandas as pd
import numpy as np
import io
import tempfile
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ExcelExtractor:
    def __init__(self):
        self.chunk_size = 1000  # Number of rows to process at once
        self.max_sheet_size = 100000  # Maximum number of cells to process per sheet

    def _process_chunk(self, chunk: pd.DataFrame, sheet_name: str, start_idx: int) -> List[str]:
        """Process a chunk of the DataFrame as CSV output only"""
        texts = []

        # Add CSV header only once at the start
        if start_idx == 0:
            texts.append(",".join(str(col) for col in chunk.columns))

        # Process rows and generate CSV
        for _, row in chunk.iterrows():
            # Convert row values to strings and handle special characters
            csv_values = []
            for value in row:
                if pd.isna(value):
                    csv_values.append('')
                else:
                    # Ensure value is converted to string
                    str_value = str(value).strip()
                    # Escape quotes and wrap in quotes if contains special characters
                    if ',' in str_value or '"' in str_value or '\n' in str_value:
                        str_value = f'"{str_value.replace("\"", "\"\"")}"'
                    csv_values.append(str_value)
            texts.append(",".join(csv_values))

        return texts

    def _process_sheet(self, sheet_data: Dict[str, Any]) -> List[str]:
        """Process a single sheet with chunking"""
        sheet_name = sheet_data['name']
        df = sheet_data['data']
        
        # Check sheet size
        total_cells = df.size
        if total_cells > self.max_sheet_size:
            logger.warning(f"Sheet '{sheet_name}' exceeds maximum size. Processing first {self.max_sheet_size} cells.")
            rows_to_process = min(len(df), self.max_sheet_size // len(df.columns))
            df = df.iloc[:rows_to_process]

        texts = []
        
        # Process DataFrame in chunks
        for start_idx in range(0, len(df), self.chunk_size):
            chunk = df.iloc[start_idx:start_idx + self.chunk_size]
            chunk_texts = self._process_chunk(chunk, sheet_name, start_idx)
            texts.extend(chunk_texts)

        return texts

    def extract_text_from_excel(self, file_bytes: bytes, file_extension: str) -> str:
        """Extract text from Excel files with optimized processing"""
        tmp_path = None
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Read Excel file with appropriate engine and optimizations
            engine = 'xlrd' if file_extension.lower() == '.xls' else 'openpyxl'
            
            # Get sheet names first to avoid loading all sheets at once
            xl = pd.ExcelFile(tmp_path, engine=engine)
            sheet_names = xl.sheet_names
            
            all_texts = []
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(
                        tmp_path,
                        sheet_name=sheet_name,
                        engine=engine,
                        na_filter=False  # Faster processing of NA values
                    )
                    sheet_data = {'name': sheet_name, 'data': df}
                    texts = self._process_sheet(sheet_data)
                    all_texts.extend(texts)
                except Exception as sheet_error:
                    logger.error(f"Error processing sheet '{sheet_name}': {sheet_error}")
                    continue

            return '\n'.join(all_texts) if all_texts else ""

        except pd.errors.EmptyDataError:
            logger.error("The Excel file appears to be empty")
            return ""
        except (pd.errors.ParserError, ValueError) as e:
            logger.error(f"Error parsing Excel file: {e}")
            raise ValueError(f"Invalid Excel file format: {str(e)}")
        except IOError as e:
            logger.error(f"IO Error while processing Excel file: {e}")
            raise IOError(f"Failed to read Excel file: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error extracting text from Excel: {e}")
            raise Exception(f"Error extracting text from Excel: {str(e)}")
        finally:
            # Clean up temporary file
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")
