import torch
import numpy as np
from pathlib import Path
from src.config import config
from src.model.detector import DeepfakeDetector
from src.data.preprocessing.video_processor import VideoProcessor
from src.data.preprocessing.audio_processor import AudioProcessor

class DetectionPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._load_model()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
    def _load_model(self):
        print(f"Loading model from {config.MODEL_PATH} ...")
        model = DeepfakeDetector()
        checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")
        return model.to(self.device)
    
    def _preprocess(self, video_data, audio_features):
        # Process video: normalize and adjust dimensions
        full_frames = torch.FloatTensor(video_data['full'] / 255.0).permute(0, 3, 1, 2)
        lip_frames = torch.FloatTensor(video_data['lip'] / 255.0).permute(0, 3, 1, 2)
        # Process audio
        audio_tensor = torch.FloatTensor(audio_features)
        
        return {
            'full': full_frames.unsqueeze(0).to(self.device),
            'lip': lip_frames.unsqueeze(0).to(self.device)
        }, audio_tensor.unsqueeze(0).to(self.device)
    
    def analyze(self, video_path: str):
        try:
            print(f"Processing video: {video_path}")
            with torch.no_grad():
                video_data = self.video_processor.process_video(video_path)
                audio_features = self.audio_processor.process_audio(video_path)
                
                video_tensor, audio_tensor = self._preprocess(video_data, audio_features)
                
                output = self.model(video_tensor, audio_tensor)
                # Since the model already applies Sigmoid, output is between 0 and 1.
                confidence = output.item()
                print(f"Confidence: {confidence}")
                
                return {
                    'fake': confidence >= config.THRESHOLD,
                    'confidence': confidence,
                    'path': video_path
                }
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                'error': str(e),
                'path': video_path
            }
