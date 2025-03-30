from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import os
import logging
from .audio_extractor import AudioExtractor

class VideoExtractor:
    def __init__(self):
        self.audio_extractor = AudioExtractor()
        self.supported_formats = ['mp4', 'avi', 'mov', 'mkv']

    def extract_audio_from_video(self, video_bytes, file_extension):
        """Extract audio from video file"""
        if not video_bytes:
            raise ValueError("No video data provided")

        format_type = file_extension[1:].lower()
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported video format: {format_type}")

        temp_files = []
        temp_dir = None
        try:
            # Create a temporary directory to store our files
            temp_dir = tempfile.mkdtemp()
            
            # Save video bytes to temporary file
            video_path = os.path.join(temp_dir, f"input.{format_type}")
            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            temp_files.append(video_path)

            # Load video and extract audio with explicit parameters
            video = VideoFileClip(video_path)
            
            # Create output WAV path
            audio_path = os.path.join(temp_dir, "output.wav")
            temp_files.append(audio_path)

            # Write audio with explicit parameters for better compatibility
            video.audio.write_audiofile(
                audio_path,
                fps=16000,       # Sample rate
                nbytes=2,        # 16-bit depth
                codec='pcm_s16le',  # PCM 16-bit encoding
                ffmpeg_params=['-ac', '1']  # Mono channel
            )
            
            # Clean up video resources
            video.close()
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            return audio_bytes
            
        except Exception as e:
            logging.error(f"Error extracting audio from video: {str(e)}")
            raise
        finally:
            # Clean up temporary files
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

    def extract_text_from_video(self, video_bytes, file_extension):
        """Extract text from video by converting to audio first"""
        try:
            # Extract audio from video
            audio_bytes = self.extract_audio_from_video(video_bytes, file_extension)
            
            # Extract text from audio
            return self.audio_extractor.extract_text_from_audio(audio_bytes, '.wav')
            
        except Exception as e:
            logging.error(f"Error extracting text from video: {str(e)}")
            raise
