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
        """Process a chunk of the DataFrame efficiently"""
        texts = []
        
        # Convert chunk to string representation without index
        chunk_text = chunk.to_string(index=False, na_rep='')
        if chunk_text.strip():
            texts.append(chunk_text)

        # Process non-empty cells with their coordinates
        for row_idx, row in chunk.iterrows():
            row_texts = []
            for col_idx, value in enumerate(row):
                if pd.notna(value) and value != '':
                    col_name = chunk.columns[col_idx]
                    actual_row = start_idx + row_idx + 1
                    row_texts.append(f"{col_name}{actual_row}: {value}")
            if row_texts:
                texts.extend(row_texts)

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

        texts = [f"\nSheet: {sheet_name}\n"]
        
        # Process DataFrame in chunks
        for start_idx in range(0, len(df), self.chunk_size):
            chunk = df.iloc[start_idx:start_idx + self.chunk_size]
            chunk_texts = self._process_chunk(chunk, sheet_name, start_idx)
            texts.extend(chunk_texts)

        return texts

    def extract_text_from_excel(self, file_bytes: bytes, file_extension: str) -> str:
        """Extract text from Excel files with optimized processing"""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # Read Excel file with appropriate engine and optimizations
                engine = 'xlrd' if file_extension.lower() == '.xls' else 'openpyxl'
                sheets_dict = pd.read_excel(
                    tmp_path,
                    sheet_name=None,  # Read all sheets
                    engine=engine,
                    na_filter=False,  # Faster processing of NA values
                )

                # Prepare sheet data for parallel processing
                sheet_data = [
                    {'name': name, 'data': df} 
                    for name, df in sheets_dict.items()
                ]

                # Process sheets in parallel
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(self._process_sheet, sheet_data))

                # Flatten results and join
                extracted_text = '\n'.join([text for texts in results for text in texts])
                return extracted_text

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")

        except Exception as e:
            logger.error(f"Error extracting text from Excel: {e}")
            raise Exception(f"Error extracting text from Excel: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Error extracting text from Excel: {str(e)}")
