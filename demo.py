import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional
import tempfile
import os
import shutil
import mediapipe as mp

from models.japanese_lipnet import JapaneseLipNet
from utils.japanese_tokenizer import JapanesePhonemeTokenizer
from config import Config, ModelConfig

class LipReadingDemo:
    def __init__(
        self, 
        model_path: str,
        config: Optional[Config] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.config = config or Config()
        self.tokenizer = JapanesePhonemeTokenizer()
        
        # Initialize face mesh detection
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Initialize model
        self.model = JapaneseLipNet(
            num_classes=self.tokenizer.num_classes,
            d_model=self.config.model.d_model,
            num_encoder_layers=self.config.model.num_encoder_layers,
            num_decoder_layers=self.config.model.num_decoder_layers,
            num_heads=self.config.model.num_heads,
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load model weights
        state_dict = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def _extract_lip_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract lip region from frame using MediaPipe Face Mesh"""
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        # Get lip landmarks
        face_landmarks = results.multi_face_landmarks[0]
        lip_landmarks = [
            face_landmarks.landmark[i] for i in range(0, 468)
            if i in mp.solutions.face_mesh.FACEMESH_LIPS
        ]
        
        # Calculate bounding box for lips
        h, w = frame.shape[:2]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        
        for landmark in lip_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            
        # Add padding
        padding = 30
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Extract and resize ROI
        roi = frame[y_min:y_max, x_min:x_max]
        roi = cv2.resize(roi, (self.config.model.img_width, self.config.model.img_height))
        return roi
        
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Process video file and return predicted text"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract lip ROI
            roi = self._extract_lip_roi(frame)
            if roi is None:
                continue
                
            # Convert to grayscale and normalize
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = roi.astype(np.float32) / 255.0
            frames.append(roi)
            
        cap.release()
        
        if not frames:
            raise ValueError("No valid frames found in the video")
            
        # Prepare input tensor
        x = np.array(frames)
        x = torch.FloatTensor(x).unsqueeze(1)  # Add channel dimension
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            logits = self.model(x)
            predictions = self.tokenizer.decode_ctc(logits[0].cpu().numpy())
            
        predicted_text = "".join(predictions)
        
        # Save visualization if output path is provided
        if output_path:
            self._save_visualization(video_path, predicted_text, output_path)
            
        return predicted_text
        
    def _save_visualization(self, video_path: str, predicted_text: str, output_path: str):
        """Save video with predicted text overlay"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        line_type = 2
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add text overlay
            cv2.putText(
                frame, 
                predicted_text,
                (50, height - 50),
                font, 
                font_scale,
                font_color,
                line_type
            )
            
            out.write(frame)
            
        cap.release()
        out.release()

def main():
    parser = argparse.ArgumentParser(description="Demo script for Japanese LipNet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_path", type=str, help="Path to output visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Initialize demo
    demo = LipReadingDemo(
        model_path=args.model_path,
        device=args.device
    )
    
    # Process video
    try:
        predicted_text = demo.process_video(args.video_path, args.output_path)
        print(f"Predicted text: {predicted_text}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()