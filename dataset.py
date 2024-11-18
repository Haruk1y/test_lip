import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from typing import Dict, Any, Tuple
import pandas as pd

class ROHANDataset(Dataset):
    def __init__(self, 
                 video_dir: str,
                 lab_dir: str,
                 landmark_dir: str,
                 file_list: str,
                 tokenizer: JapanesePhonemeTokenizer,
                 vid_padding: int = 200,
                 txt_padding: int = 200,
                 phase: str = 'train'):
        """
        Args:
            video_dir: 動画ファイルのディレクトリ
            lab_dir: LABファイルのディレクトリ
            landmark_dir: ランドマークCSVのディレクトリ
            file_list: ファイルリスト
            tokenizer: 音素トークナイザー
            vid_padding: ビデオパディング長
            txt_padding: テキストパディング長
            phase: 'train' or 'val'
        """
        self.video_dir = video_dir
        self.lab_dir = lab_dir
        self.landmark_dir = landmark_dir
        self.tokenizer = tokenizer
        self.vid_padding = vid_padding
        self.txt_padding = txt_padding
        self.phase = phase
        
        # ファイルリストの読み込み
        with open(file_list, 'r') as f:
            self.files = [line.strip() for line in f.readlines()]
            
    def __len__(self) -> int:
        return len(self.files)
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """動画の読み込みと前処理"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # グレースケール変換
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # リサイズ (128x64に統一)
            frame = cv2.resize(frame, (128, 64))
            frames.append(frame)
        cap.release()
        
        return np.array(frames)
    
    def _load_landmarks(self, csv_path: str) -> np.ndarray:
        """ランドマークの読み込み"""
        df = pd.read_csv(csv_path)
        return df.values
    
    def _align_phonemes(self, phoneme_info: List[Dict], video_frames: int,
                       fps: float = 29.95) -> List[str]:
        """音素を動画フレームに合わせてアライメント"""
        frame_phonemes = []
        current_frame = 0
        
        for frame_idx in range(video_frames):
            time_ns = int(frame_idx * (1e9 / fps))  # フレーム番号を時間(ns)に変換
            
            # 現在の時刻に対応する音素を探す
            current_phoneme = 'sil'  # デフォルトは無音
            for info in phoneme_info:
                if info['start'] <= time_ns <= info['end']:
                    current_phoneme = info['phoneme']
                    break
            frame_phonemes.append(current_phoneme)
            
        return frame_phonemes
    
    def _padding(self, array: np.ndarray, length: int) -> np.ndarray:
        """配列をパディング"""
        if array.shape[0] > length:
            return array[:length]
        else:
            pad_width = [(0, length - array.shape[0])] + \
                       [(0, 0) for _ in range(len(array.shape) - 1)]
            return np.pad(array, pad_width, mode='constant')
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_id = self.files[idx]
        
        # 各ファイルパスの構築
        video_path = os.path.join(self.video_dir, f"LFROI_{file_id}.mp4")
        lab_path = os.path.join(self.lab_dir, f"{file_id}.lab")
        landmark_path = os.path.join(self.landmark_dir, f"{file_id}_points.csv")
        
        # データの読み込み
        video = self._load_video(video_path)
        phoneme_info = self.tokenizer.parse_lab_file(lab_path)
        landmarks = self._load_landmarks(landmark_path)
        
        # 音素のアライメント
        frame_phonemes = self._align_phonemes(phoneme_info, len(video))
        phoneme_indices = self.tokenizer.encode(frame_phonemes)
        
        # パディング
        video = self._padding(video, self.vid_padding)
        phoneme_indices = self._padding(phoneme_indices, self.txt_padding)
        
        # データ拡張（学習時のみ）
        if self.phase == 'train':
            if np.random.random() < 0.5:
                video = video[:, :, ::-1]  # 水平反転
        
        # チャンネル次元の追加とテンソル変換
        video = np.expand_dims(video, axis=1)  # (T, 1, H, W)
        video = video / 255.0  # 正規化
        
        return {
            'vid': torch.FloatTensor(video),
            'txt': torch.LongTensor(phoneme_indices),
            'vid_len': torch.LongTensor([len(video)]),
            'txt_len': torch.LongTensor([len(frame_phonemes)])
        }