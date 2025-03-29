from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import os
from .audio_extractor import AudioExtractor

class VideoExtractor:
    def __init__(self):
        self.audio_extractor = AudioExtractor()

    def extract_audio_from_video(self, video_bytes, file_extension):
        """Extract audio from video file"""
        try:
            # Save video bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(video_bytes)
                video_path = tmp.name

            # Load video and extract audio
            video = VideoFileClip(video_path)
            audio_path = video_path + '.wav'
            video.audio.write_audiofile(audio_path)
            
            # Clean up video file
            video.close()
            os.unlink(video_path)
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Clean up audio file
            os.unlink(audio_path)
            
            return audio_bytes
            
        except Exception as e:
            raise Exception(f"Error extracting audio from video: {str(e)}")

    def extract_text_from_video(self, video_bytes, file_extension):
        """Extract text from video by converting to audio first"""
        try:
            # Extract audio from video
            audio_bytes = self.extract_audio_from_video(video_bytes, file_extension)
            
            # Extract text from audio
            return self.audio_extractor.extract_text_from_audio(audio_bytes, '.wav')
            
        except Exception as e:
            raise Exception(f"Error extracting text from video: {str(e)}")
