import librosa
import numpy as np
import warnings
from src.config import config

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioProcessor:
    def process_audio(self, video_path: str):
        target_length = int(config.SAMPLE_RATE * config.AUDIO_LENGTH)
        
        try:
            y, sr = librosa.load(
                video_path,
                sr=config.SAMPLE_RATE,
                duration=config.AUDIO_LENGTH,
                res_type="kaiser_fast",
                mono=True
            )
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc = librosa.util.fix_length(mfcc, size=target_length, axis=1)
            return mfcc
            
        except Exception as e:
            return np.zeros((40, target_length), dtype=np.float32)
