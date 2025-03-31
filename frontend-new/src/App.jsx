import { useState, useCallback, useMemo, lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from '@emotion/styled';
import './App.css';

// Styled components (no changes needed here)
const AppContainer = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1.5rem;
  color: #2d3748;
  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

const Card = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 24px;
  padding: 2.5rem;
  width: 95%;
  max-width: 900px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  position: relative;
  overflow: hidden;
  margin: 0 auto;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
  }

  @media (max-width: 768px) {
    padding: 1.5rem;
    gap: 1rem;
    width: 100%;
  }
`;

const Logo = styled(motion.div)`
  text-align: center;

  h1 {
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 800;
    background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    line-height: 1.2;
  }

  p {
    font-size: clamp(0.9rem, 2vw, 1.1rem);
    color: #4a5568;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.5;
  }

  @media (max-width: 768px) {
    h1 {
      margin-bottom: 0.25rem;
    }
    p {
      font-size: 0.9rem;
    }
  }
`;

const UploadZone = styled(motion.div)`
  border: 2px dashed rgba(79, 172, 254, 0.4);
  border-radius: 16px;
  padding: clamp(1.5rem, 4vw, 3rem) 1.5rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.5);
  
  &:hover {
    border-color: #4facfe;
    background: rgba(79, 172, 254, 0.1);
  }

  h3 {
    font-size: clamp(1.2rem, 3vw, 1.5rem);
    color: #2d3748;
    margin-bottom: 0.75rem;
  }

  p {
    color: #4a5568;
    margin: 0.5rem 0;
    font-size: clamp(0.875rem, 2vw, 1rem);
  }

  .file-info {
    margin-top: 0.75rem;
    font-weight: 500;
    color: #4facfe;
  }

  .file-type {
    color: #718096;
    font-size: 0.9rem;
    margin-top: 0.25rem;
  }

  .format-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
    padding: 0 1rem;

    span {
      background: rgba(255, 255, 255, 0.5);
      padding: 0.5rem;
      border-radius: 8px;
      font-size: 0.9rem;
      color: #4a5568;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: all 0.2s ease;

      &:hover {
        background: rgba(79, 172, 254, 0.1);
        transform: translateY(-1px);
      }
    }
  }

  @media (max-width: 768px) {
    padding: 1.25rem;
    h3 {
      margin-bottom: 0.5rem;
    }
    p {
      margin: 0.25rem 0;
    }
    .format-grid {
      grid-template-columns: 1fr;
      padding: 0;
    }
  }
`;

const Button = styled(motion.button)`
  background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
  color: white;
  border: none;
  padding: clamp(0.75rem, 2vw, 1rem) clamp(1.5rem, 4vw, 2.5rem);
  border-radius: 30px;
  font-size: clamp(1rem, 2.5vw, 1.1rem);
  font-weight: 600;
  cursor: pointer;
  width: auto;
  margin: 0.5rem auto;
  display: block;
  min-width: clamp(180px, 50%, 200px);
  box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
  transition: all 0.3s ease;
  
  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
    box-shadow: none;
  }

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  @media (max-width: 768px) {
    width: 100%;
    margin: 0.25rem auto;
  }
`;

const ResultContainer = styled(motion.div)`
  background: rgba(255, 255, 255, 0.8);
  border-radius: 16px;
  padding: 1.5rem;
  max-height: 400px;
  overflow-y: auto;
  box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.05);

  h3 {
    color: #2d3748;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  pre {
    white-space: pre-wrap;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.9rem;
    color: #4a5568;
    padding: 0.5rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.5);
    overflow-x: auto;
  }

  @media (max-width: 768px) {
    padding: 1rem;
    max-height: 300px;
    
    h3 {
      font-size: 1.1rem;
      margin-bottom: 0.75rem;
    }
    
    pre {
      font-size: 0.8rem;
    }
  }
`;

const RequirementsContainer = styled(motion.div)`
  background: rgba(235, 250, 255, 0.9);
  border-radius: 16px;
  padding: 1.5rem;
  max-height: 500px;
  overflow-y: auto;
  box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.05);
  border-left: 4px solid #4facfe;

  h3 {
    color: #2d3748;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .requirements-text {
    white-space: pre-wrap;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.9rem;
    color: #2d3748;
    padding: 0.5rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.7);
    overflow-x: auto;
    line-height: 1.6;
  }

  @media (max-width: 768px) {
    padding: 1rem;
    max-height: 350px;
  }
`;

const FormatSelector = styled.div`
  display: none;
`;

const ErrorMessage = styled(motion.div)`
  background: rgba(255, 59, 48, 0.1);
  border-radius: 12px;
  padding: 1rem 1.5rem;
  margin: 1rem 0;
  color: #dc2626;
  text-align: center;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;

  svg {
    width: 20px;
    height: 20px;
  }
`;

// Loading spinner component
const Spinner = styled(motion.div)`
  border: 3px solid rgba(79, 172, 254, 0.2);
  border-top: 3px solid #4facfe;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  margin-right: 10px;
  display: inline-block;
`;

// Memoized file type message function
const getFileTypeMessage = (filename) => {
  const ext = filename.toLowerCase().split('.').pop();
  const types = {
    // Document formats
    pdf: 'PDF document - Extracting text with enhanced OCR',
    docx: 'Word document - Extracting formatted text',
    txt: 'Text file - Direct text extraction',
    xlsx: 'Excel file - Extracting spreadsheet data',
    xls: 'Excel file - Extracting spreadsheet data',
    eml: 'Email file - Parsing email content',
    html: 'Web page - Extracting formatted content',
    htm: 'Web page - Extracting formatted content',
    csv: 'CSV file - Extracting spreadsheet data',
    // Audio formats
    mp3: 'Audio file - Converting speech to text',
    wav: 'Audio file - Converting speech to text',
    m4a: 'Audio file - Converting speech to text',
    ogg: 'Audio file - Converting speech to text',
    // Video formats
    mp4: 'Video file - Extracting audio and converting to text',
    avi: 'Video file - Extracting audio and converting to text',
    mov: 'Video file - Extracting audio and converting to text',
    mkv: 'Video file - Extracting audio and converting to text',
    // Image formats
    png: 'Image file - Using advanced OCR with EasyOCR',
    jpg: 'Image file - Using advanced OCR with EasyOCR',
    jpeg: 'Image file - Using advanced OCR with EasyOCR',
    tiff: 'Image file - Using advanced OCR with EasyOCR',
    bmp: 'Image file - Using advanced OCR with EasyOCR',
    gif: 'Image file - Using advanced OCR with EasyOCR',
    webp: 'Image file - Using advanced OCR with EasyOCR'
  };
  return types[ext] || 'Processing file...';
};

// API service with retry logic
const API_URL = 'http://localhost:8000';

const apiService = {
  extractText: async (formData, onProgress) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      throw new Error('Request timed out. Please try again.');
    }, 60000); // 60s timeout
    
    try {
      const response = await fetch(`${API_URL}/extract-text`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          detail: `Server error: ${response.status} ${response.statusText}`
        }));
        throw new Error(errorData.detail || 'Failed to extract text');
      }
      
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Invalid response format from server');
      }
      
      const data = await response.json();
      if (!data.extracted_text) {
        throw new Error('No text content in response');
      }
      
      return data.extracted_text;
    } finally {
      clearTimeout(timeoutId);
    }
  },
  
  // New method for the extract-and-generate endpoint
  extractAndGenerate: async (formData, formatType, onProgress) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      throw new Error('Request timed out. Please try again.');
    }, 120000); // 120s timeout (longer for this operation)
    
    try {
      const response = await fetch(`${API_URL}/extract-and-generate?format_type=${formatType}`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          detail: `Server error: ${response.status} ${response.statusText}`
        }));
        throw new Error(errorData.detail || 'Failed to process file');
      }
      
      const data = await response.json();
      return data;
    } finally {
      clearTimeout(timeoutId);
    }
  },

  // Method to generate requirements from already extracted text
  generateRequirements: async (text, formatType) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      throw new Error('Request timed out. Please try again.');
    }, 60000); // 60s timeout
    
    try {
      const response = await fetch(`${API_URL}/generate-requirements`, {
        method: 'POST',
        body: JSON.stringify({
          text: text,
          format_type: formatType
        }),
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          detail: `Server error: ${response.status} ${response.statusText}`
        }));
        throw new Error(errorData.detail || 'Failed to generate requirements');
      }
      
      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  },
  
  // Retry logic
  retry: async (fn, retries = 2, delay = 1000) => {
    try {
      return await fn();
    } catch (error) {
      if (retries <= 0) throw error;
      await new Promise(resolve => setTimeout(resolve, delay));
      return apiService.retry(fn, retries - 1, delay * 1.5);
    }
  }
};

// File validation
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const ALLOWED_EXTENSIONS = [
  // Document formats
  '.xlsx', '.xls', '.csv', '.pdf', '.doc', '.docx', '.txt', '.eml', '.html', '.htm',
  // Audio formats
  '.mp3', '.wav', '.m4a', '.ogg',
  // Video formats
  '.mp4', '.avi', '.mov', '.mkv',
  // Image formats
  '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp'
];

const validateFile = (file) => {
  if (!file) return 'Please select a file';
  if (file.size > MAX_FILE_SIZE) return 'File size exceeds 50MB limit';
  
  const extension = '.' + file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    return `Unsupported file type. Allowed types: ${ALLOWED_EXTENSIONS.join(', ')}`;
  }
  return null;
};

// Main App component
function App() {
  const [file, setFile] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [requirements, setRequirements] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState(0);
  const [formatType, setFormatType] = useState('standard');
  const [processingStep, setProcessingStep] = useState('');

  // Memoized file validation
  const fileValidationError = useMemo(() => {
    return file ? validateFile(file) : null;
  }, [file]);

  // Optimized event handlers with useCallback
  const handleFileChange = useCallback((e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setExtractedText('');
      setRequirements('');
      setError('');
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setExtractedText('');
      setRequirements('');
      setError('');
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const handleGenerateRequirements = useCallback(async () => {
    if (!extractedText || extractedText.trim().length < 10) {
      setError('Extracted text is too short to generate requirements');
      return;
    }

    setLoading(true);
    setError('');
    setProcessingStep('Generating requirements...');
    
    try {
      const result = await apiService.retry(() => 
        apiService.generateRequirements(extractedText, formatType)
      );
      
      if (result && result.requirements) {
        setRequirements(result.requirements);
      } else {
        throw new Error('Failed to generate requirements');
      }
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Error generating requirements. Please try again.');
    } finally {
      setLoading(false);
      setProcessingStep('');
    }
  }, [extractedText, formatType]);

  const handleExtractAndGenerate = useCallback(async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    if (fileValidationError) {
      setError(fileValidationError);
      return;
    }

    setLoading(true);
    setError('');
    setExtractedText('');
    setRequirements('');
    setProgress(0);
    setProcessingStep('Extracting text...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use the extract-and-generate endpoint
      const result = await apiService.retry(() => 
        apiService.extractAndGenerate(formData, formatType, (progress) => {
          setProgress(progress);
        })
      );
      
      setExtractedText(result.extracted_text || '');
      setRequirements(result.requirements || '');
      
      if (!result.requirements && result.warning) {
        setError(result.warning);
      }
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Error processing file. Please try again.');
    } finally {
      setLoading(false);
      setProgress(0);
      setProcessingStep('');
    }
  }, [file, fileValidationError, formatType]);

  // Memoized file type message
  const fileTypeMessage = useMemo(() => {
    return file ? getFileTypeMessage(file.name) : '';
  }, [file]);

  return (
    <AppContainer>
      <Card
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Logo>
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            Requirements Generator
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            Extract text and generate software requirements
          </motion.p>
        </Logo>
        
        <UploadZone
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => document.getElementById('fileInput').click()}
        >
          <input
            type="file"
            id="fileInput"
            onChange={handleFileChange}
            accept=".pdf,.png,.jpg,.jpeg,.docx,.txt,.xlsx,.xls,.eml,.html,.mp3,.wav,.mp4,.avi,.mov"
            style={{ display: 'none' }}
          />
          <h3>Upload Your File</h3>
          {file ? (
            <>
              <p className="file-info">Selected: {file.name}</p>
              <p className="file-type">{fileTypeMessage}</p>
            </>
          ) : (
            <>
              <p>Drop your file here or click to browse</p>
              <p>Supports multiple formats:</p>
              <div className="format-grid">
                <span>📄 Documents (PDF, Word, TXT)</span>
                <span>📊 Excel Sheets</span>
                <span>📧 Emails (EML)</span>
                <span>🖼️ Images (PNG, JPG)</span>
                <span>🎥 Videos (MP4, AVI)</span>
                <span>🎵 Audio (MP3, WAV)</span>
                <span>🌐 Web Pages (HTML)</span>
              </div>
            </>
          )}
        </UploadZone>

        <Button
          onClick={handleExtractAndGenerate}
          disabled={!file || loading}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {loading ? (
            <>
              <Spinner animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }} />
              {processingStep || 'Processing...'}
            </>
          ) : 'Extract and Generate Requirements'}
        </Button>

        <AnimatePresence>
          {error && (
            <ErrorMessage
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              {error}
            </ErrorMessage>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {extractedText && (
            <ResultContainer
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <h3>Extracted Text:</h3>
              <pre>{extractedText}</pre>
              
              {!requirements && !loading && (
                <Button 
                  onClick={handleGenerateRequirements}
                  disabled={loading}
                  style={{ marginTop: '1rem' }}
                >
                  Generate Requirements
                </Button>
              )}
            </ResultContainer>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {requirements && (
            <RequirementsContainer
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <h3>Generated Requirements:</h3>
              <div className="requirements-text">{requirements}</div>
            </RequirementsContainer>
          )}
        </AnimatePresence>
      </Card>
    </AppContainer>
  );
}

export default App;
