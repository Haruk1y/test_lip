import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any
import cv2

class ROHANDataset(Dataset):
    def __init__(self,
                 video_dir: str,
                 lab_dir: str,
                 landmark_dir: str,
                 file_list: str,
                 tokenizer,
                 vid_padding: int = 459,  # ROHANの最大フレーム数
                 txt_padding: int = 459,
                 phase: str = 'train',
                 target_size: tuple = (96, 96)):
        """
        Args:
            video_dir: 処理済み動画データのディレクトリ
            lab_dir: LABファイルのディレクトリ
            landmark_dir: ランドマークデータのディレクトリ
            file_list: ファイルリスト
            tokenizer: 音素トークナイザー
            vid_padding: ビデオパディング長
            txt_padding: テキストパディング長
            phase: 'train' or 'val' or 'test'
        """
        self.video_dir = Path(video_dir)
        self.lab_dir = Path(lab_dir)
        self.landmark_dir = Path(landmark_dir)
        self.tokenizer = tokenizer
        self.vid_padding = vid_padding
        self.txt_padding = txt_padding
        self.phase = phase
        self.target_size = target_size
        
        # ファイルリストの読み込み
        with open(file_list, 'r') as f:
            self.files = [line.strip() for line in f.readlines()]

        # npzファイルのディレクトリ（data/train/やdata/val/）
        self.npz_dir = Path("data") / phase
    
    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """フレームのリサイズ処理"""
        h, w = self.target_size
        resized = []
        for frame in frames:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            resized.append(frame)
        return np.array(resized)

    def __len__(self) -> int:
        return len(self.files)
    
    def _load_data(self, npz_path: Path) -> Dict[str, np.ndarray]:
        """処理済みデータの読み込み"""
        data = np.load(npz_path)
        return {key: data[key] for key in data.files}
    
    def _padding(self, array: np.ndarray, length: int) -> np.ndarray:
        """配列をパディング"""
        if array.shape[0] > length:
            return array[:length]
        else:
            pad_width = [(0, length - array.shape[0])]
            if len(array.shape) > 1:
                pad_width.extend((0, 0) for _ in range(len(array.shape) - 1))
            return np.pad(array, pad_width, mode='constant', constant_values=0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_id = self.files[idx]
        npz_path = self.npz_dir / f"{file_id}.npz"
        
        try:
            data = self._load_data(npz_path)
        except FileNotFoundError:
            print(f"File not found: {npz_path}")
            raise
        
        frames = data['frames']  # (T, H, W)
        phonemes = data['phonemes']
        
         # フレームのリサイズ
        frames = self._resize_frames(frames)

        # メモリ効率の良い処理
        frames = self._padding(frames, self.vid_padding)
        phoneme_indices = self.tokenizer.encode(phonemes)
        phoneme_indices = self._padding(phoneme_indices, self.txt_padding)

        # パディング後の長さを確認
        if len(phoneme_indices) > self.txt_padding:
            print(f"Warning: Sequence length {len(phoneme_indices)} exceeds padding {self.txt_padding}")
        
        if self.phase == 'train' and np.random.random() < 0.5:
            frames = np.ascontiguousarray(frames[:, ::-1])
        
        ## 正規化とチャンネル次元の追加
        frames = frames.astype(np.float32) / 255.0
        frames = np.expand_dims(frames, axis=0)  # (1, T, H, W)
        
        return {
            'vid': torch.from_numpy(frames).float(),  # shape: (1, T, H, W)
            'txt': torch.from_numpy(phoneme_indices).long(),
            'vid_len': torch.tensor([frames.shape[1]], dtype=torch.long),
            'txt_len': torch.tensor([len(phonemes)], dtype=torch.long)
        }