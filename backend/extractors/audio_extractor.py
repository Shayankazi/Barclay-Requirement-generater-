import speech_recognition as sr
import whisper
from pydub import AudioSegment
import io
import tempfile
import os
import logging

class AudioExtractor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.model = whisper.load_model("base")
        self.supported_formats = ['mp3', 'wav', 'ogg', 'm4a']

    def convert_to_wav(self, audio_bytes, original_format):
        """Convert audio to WAV format"""
        if original_format not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {original_format}")

        temp_files = []
        try:
            # Create a temporary directory to store our files
            temp_dir = tempfile.mkdtemp()
            
            # Save bytes to temporary file with original format
            input_path = os.path.join(temp_dir, f"input.{original_format}")
            with open(input_path, 'wb') as f:
                f.write(audio_bytes)
            temp_files.append(input_path)

            # Convert to WAV with explicit parameters
            audio = AudioSegment.from_file(input_path, format=original_format)
            
            # Create output WAV path
            wav_path = os.path.join(temp_dir, "output.wav")
            temp_files.append(wav_path)
            
            # Export with explicit parameters for better compatibility
            audio.export(
                wav_path,
                format='wav',
                parameters=[
                    '-ar', '16000',  # Sample rate
                    '-ac', '1',      # Mono channel
                    '-acodec', 'pcm_s16le'  # PCM 16-bit encoding
                ]
            )
            
            return wav_path, temp_files
            
        except Exception as e:
            # Clean up any files if there's an error
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            logging.error(f"Error converting {original_format} to WAV: {str(e)}")
            raise

    def extract_text_whisper(self, audio_path):
        """Extract text using OpenAI's Whisper"""
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                language='en',  # Explicitly set language
                task='transcribe'
            )
            return result["text"].strip()
            
        except Exception as e:
            logging.error(f"Error extracting text with Whisper: {str(e)}")
            raise

    def extract_text_sphinx(self, audio_path):
        """Extract text using CMU Sphinx"""
        try:
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.record(source)
                return self.recognizer.recognize_sphinx(audio).strip()
                
        except Exception as e:
            logging.error(f"Error extracting text with Sphinx: {str(e)}")
            raise

    def extract_text_from_audio(self, audio_bytes, file_extension):
        """Main method to extract text from audio"""
        if not audio_bytes:
            raise ValueError("No audio data provided")

        temp_files = []
        temp_dir = None
        try:
            # Remove the dot from extension and convert to lowercase
            format_type = file_extension[1:].lower()
            
            # Convert to WAV if needed
            wav_path, new_temp_files = self.convert_to_wav(audio_bytes, format_type)
            temp_files.extend(new_temp_files)
            
            try:
                # Try Whisper first
                text = self.extract_text_whisper(wav_path)
            except Exception as whisper_error:
                logging.warning(f"Whisper failed, falling back to Sphinx: {str(whisper_error)}")
                # Fallback to Sphinx
                text = self.extract_text_sphinx(wav_path)
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"Error extracting text from audio: {str(e)}")
            raise
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logging.warning(f"Failed to clean up temporary file {temp_file}: {str(e)}")
            
            # Clean up temporary directory if it exists
            if temp_dir and os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except Exception as e:
                    logging.warning(f"Failed to clean up temporary directory: {str(e)}")
