import speech_recognition as sr
import whisper
from pydub import AudioSegment
import io
import tempfile
import os

class AudioExtractor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.model = whisper.load_model("base")

    def convert_to_wav(self, audio_bytes, original_format):
        """Convert audio to WAV format"""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{original_format}') as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Convert to WAV
            audio = AudioSegment.from_file(tmp_path, format=original_format)
            wav_path = tmp_path + '.wav'
            audio.export(wav_path, format='wav')
            
            # Clean up original temp file
            os.unlink(tmp_path)
            
            return wav_path
            
        except Exception as e:
            raise Exception(f"Error converting audio to WAV: {str(e)}")

    def extract_text_whisper(self, audio_path):
        """Extract text using OpenAI's Whisper"""
        try:
            # Transcribe audio
            result = self.model.transcribe(audio_path)
            return result["text"]
            
        except Exception as e:
            raise Exception(f"Error extracting text with Whisper: {str(e)}")

    def extract_text_sphinx(self, audio_path):
        """Extract text using CMU Sphinx"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                return self.recognizer.recognize_sphinx(audio)
                
        except Exception as e:
            raise Exception(f"Error extracting text with Sphinx: {str(e)}")

    def extract_text_from_audio(self, audio_bytes, file_extension):
        """Main method to extract text from audio"""
        try:
            # Remove the dot from extension
            format_type = file_extension[1:]
            
            # Convert to WAV if needed
            wav_path = self.convert_to_wav(audio_bytes, format_type)
            
            try:
                # Try Whisper first
                text = self.extract_text_whisper(wav_path)
            except Exception:
                # Fallback to Sphinx
                text = self.extract_text_sphinx(wav_path)
            
            # Clean up temporary WAV file
            os.unlink(wav_path)
            
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from audio: {str(e)}")
