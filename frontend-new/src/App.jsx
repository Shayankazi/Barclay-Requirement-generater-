import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from '@emotion/styled';
import './App.css';

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
  padding: 2rem;
  margin-top: 1rem;
  max-height: 400px;
  overflow-y: auto;
  font-family: 'Courier New', monospace;
  position: relative;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);

  h3 {
    font-size: 1.2rem;
    color: #2d3748;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  pre {
    text-align: left;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.6;
    color: #4a5568;
    background: rgba(255, 255, 255, 0.5);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid rgba(79, 172, 254, 0.2);
  }
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

const getFileTypeMessage = (filename) => {
  const ext = filename.toLowerCase().split('.').pop();
  const types = {
    pdf: 'PDF document - Extracting text with enhanced OCR',
    docx: 'Word document - Extracting formatted text',
    txt: 'Text file - Direct text extraction',
    png: 'Image file - Using advanced OCR with EasyOCR',
    jpg: 'Image file - Using advanced OCR with EasyOCR',
    jpeg: 'Image file - Using advanced OCR with EasyOCR',
    xlsx: 'Excel file - Extracting spreadsheet data',
    xls: 'Excel file - Extracting spreadsheet data',
    eml: 'Email file - Parsing email content',
    html: 'Web page - Extracting formatted content',
    mp3: 'Audio file - Converting speech to text',
    wav: 'Audio file - Converting speech to text',
    mp4: 'Video file - Extracting audio and converting to text',
    avi: 'Video file - Extracting audio and converting to text',
    mov: 'Video file - Extracting audio and converting to text'
  };
  return types[ext] || 'Processing file...';
};

function App() {
  const [file, setFile] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setExtractedText('');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
    setExtractedText('');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const [error, setError] = useState('');

  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
  const ALLOWED_EXTENSIONS = ['.xlsx', '.xls', '.csv', '.pdf', '.doc', '.docx', '.txt', '.eml', '.html', '.htm'];
  
  const validateFile = (file) => {
    if (!file) return 'Please select a file';
    if (file.size > MAX_FILE_SIZE) return 'File size exceeds 50MB limit';
    
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      return `Unsupported file type. Allowed types: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }
    return null;
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    setError('');
    setExtractedText('');

    const formData = new FormData();
    formData.append('file', file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
      setError('Request timed out. Please try again.');
      setLoading(false);
    }, 30000); // 30s timeout

    try {
      const response = await fetch('http://localhost:8000/extract-text', {
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

      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }

      try {
        const data = JSON.parse(text);
        if (!data.extracted_text) {
          throw new Error('No text content in response');
        }
        setExtractedText(data.extracted_text);
      } catch (parseError) {
        throw new Error('Invalid JSON response from server');
      }
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Error extracting text. Please try again.');
      setExtractedText('');
    } finally {
      setLoading(false);
    }
  };

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
            TextractPro
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            Extract text from any document in seconds
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
              <p className="file-type">{getFileTypeMessage(file.name)}</p>
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
          onClick={handleUpload}
          disabled={!file || loading}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {loading ? 'Processing...' : 'Extract Text'}
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
            </ResultContainer>
          )}
        </AnimatePresence>
      </Card>
    </AppContainer>
  );
}

export default App;
