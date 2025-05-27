import cv2
import numpy as np
from src.config import config

class VideoProcessor:
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        full_frames = []
        lip_frames = []
        frame_count = 0
        lip_size = config.LIP_CROP_SIZE
        
        while cap.isOpened() and frame_count < config.FRAME_COUNT:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize full frame
            full_frame = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
            full_frames.append(full_frame)
            
            # Extract lip region (adjusting based on assumed mouth position)
            h, w = full_frame.shape[:2]
            x = max((w - lip_size) // 2, 0)
            y = max(h - lip_size - 20, 0)
            lip_frame = full_frame[y:y+lip_size, x:x+lip_size]
            lip_frames.append(lip_frame)
            
            frame_count += 1
        
        cap.release()
        
        # Padding if fewer frames than expected
        while len(full_frames) < config.FRAME_COUNT:
            full_frames.append(np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8))
            lip_frames.append(np.zeros((lip_size, lip_size, 3), dtype=np.uint8))
            
        return {
            'full': np.array(full_frames),
            'lip': np.array(lip_frames)
        }
