import os

class Config:
    # Dataset paths
    RAW_DATA_DIR = os.path.join('data', 'Datasets')
    PREPROCESSED_DIR = os.path.join('data', 'preprocessed')
    
    # Model paths
    MODEL_DIR = os.path.join('data', 'models')
    CHECKPOINT_DIR = os.path.join('data', 'checkpoints')
    
    # Inference paths
    INPUT_VIDEOS_DIR = os.path.join('data', 'input_videos')
    OUTPUT_RESULTS_DIR = os.path.join('data', 'output_results')
    
    # Video processing parameters
    FRAME_COUNT = 20
    IMG_SIZE = 112
    LIP_CROP_SIZE = 64
    
    # Audio processing parameters
    AUDIO_LENGTH = 1.0
    SAMPLE_RATE = 16000
    
    # Model architecture
    VISUAL_FEATURES = 512    # ResNet-34 features dimension remains the same
    AUDIO_FEATURES = 256     # Audio encoder output dimension remains the same
    LSTM_UNITS = 256       # Increased from 128 to 256 for better temporal modeling
    
    # Training parameters
    BATCH_SIZE = 32       # Increased from 32 to 64 (max allowed)
    LR = 5e-5                # Increased from 3e-5 to 5e-5
    EPOCHS = 30              # Increased from 30 to 50 epochs (with early stopping)
    SEED = 42
    TRAIN_RATIO = 0.8
    THRESHOLD = 0.75

    @property
    def MODEL_PATH(self):
        return os.path.join(self.MODEL_DIR, 'final_model.pth')

config = Config()
