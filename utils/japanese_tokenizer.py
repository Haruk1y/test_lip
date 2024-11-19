import numpy as np
from typing import List, Dict, Optional
import torch

class JapanesePhonemeTokenizer:
    def __init__(self):
        # 音素リスト (論文のデータに基づく)
        self.phonemes = ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 
                        'e', 'f', 'fy', 'g', 'gw', 'gy', 'h', 'hy', 'i', 'j', 
                        'k', 'kw', 'ky', 'm', 'my', 'n', 'ny', 'o', 'p', 'pau',
                        'py', 'r', 'ry', 's', 'sh', 'sil', 't', 'ts', 'ty', 'u',
                        'v', 'w', 'y', 'z']
        
        # 音素からインデックスへのマッピング
        self.phoneme2idx = {p: i for i, p in enumerate(self.phonemes)}
        self.idx2phoneme = {i: p for i, p in enumerate(self.phonemes)}
        
        # 特殊トークン
        self.pad_idx = len(self.phonemes)  # パディング用
        self.blank_idx = self.pad_idx + 1  # CTCブランク用
    
    def encode(self, phoneme_sequence: List[str]) -> np.ndarray:
        """音素列をインデックス列に変換"""
        return np.array([self.phoneme2idx[p] for p in phoneme_sequence])
    
    def decode(self, index_sequence: np.ndarray) -> List[str]:
        """インデックス列を音素列に変換"""
        return [self.idx2phoneme[i] for i in index_sequence if i < self.pad_idx]
    
    def decode_ctc(self, logits) -> List[str]:
        """CTCデコーディング (最大確率の音素を選択)"""
        # GPU テンソルを CPU に移動してから numpy に変換
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        
        # バッチの最初のサンプルを取得（バッチサイズ1の場合）
        if len(logits.shape) == 3:
            logits = logits[0]  # (T, C)
            
        pred_indices = np.argmax(logits, axis=-1)  # (T,)
        decoded = []
        prev = -1
        
        # 各時間ステップの予測を処理
        for t in range(len(pred_indices)):
            idx = pred_indices[t]
            if (idx != prev and 
                idx != self.blank_idx and 
                idx < self.pad_idx):
                decoded.append(self.idx2phoneme[idx])
            prev = idx
            
        return decoded
    
    @property
    def num_classes(self) -> int:
        """音素数 + パディング + CTCブランク"""
        return len(self.phonemes) + 2