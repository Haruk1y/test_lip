import editdistance
from typing import List, Dict, Set
import numpy as np

def compute_error_rate(predictions: List[str], targets: List[str], level: str = 'phoneme') -> float:
    """
    エラー率を計算する関数
    Args:
        predictions: 予測された音素/モーラ列のリスト
        targets: 正解の音素/モーラ列のリスト
        level: 'phoneme' または 'mora'
    Returns:
        float: エラー率
    """
    metrics = JapaneseLipReadingMetrics()
    if level == 'phoneme':
        return metrics.compute_phoneme_error_rate(predictions, targets)
    elif level == 'mora':
        return metrics.compute_mora_error_rate(predictions, targets)
    else:
        raise ValueError(f"Unknown level: {level}")

class JapaneseLipReadingMetrics:
    def __init__(self):
        # モーラ変換用のルール
        self.mora_rules = {
            # 長音
            'a': {'a': 'aa'},
            'i': {'i': 'ii'},
            'u': {'u': 'uu'},
            'e': {'e': 'ee'},
            'o': {'o': 'oo'},
            
            # 撥音
            'N': {'': 'N'},
            
            # 促音
            'cl': {'': 'Q'},
            
            # 拗音
            'y': {'a': 'ya', 'u': 'yu', 'o': 'yo'},
            
            # 特殊な音素の組み合わせ
            'ch': {'a': 'cha', 'i': 'chi', 'u': 'chu', 'e': 'che', 'o': 'cho'},
            'sh': {'a': 'sha', 'i': 'shi', 'u': 'shu', 'e': 'she', 'o': 'sho'},
            'ts': {'u': 'tsu'},
        }
        
        # 特殊トークン
        self.special_tokens = {'sil', 'pau'}
    
    def compute_phoneme_error_rate(self, predictions: List[str], targets: List[str]) -> float:
        """音素単位のエラー率を計算"""
        total_distance = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # 音素列への分割
            pred_phones = [p for p in pred if p not in self.special_tokens]
            target_phones = [p for p in target if p not in self.special_tokens]
            
            # 編集距離の計算
            distance = editdistance.eval(pred_phones, target_phones)
            
            total_distance += distance
            total_length += len(target_phones)
        
        return total_distance / total_length if total_length > 0 else 1.0
    
    def _convert_to_mora(self, phoneme_seq: List[str]) -> List[str]:
        """音素列をモーラ列に変換"""
        moras = []
        i = 0
        
        while i < len(phoneme_seq):
            if phoneme_seq[i] in self.special_tokens:
                moras.append(phoneme_seq[i])
                i += 1
                continue
                
            # 2音素の組み合わせをチェック
            if i + 1 < len(phoneme_seq):
                phoneme_pair = phoneme_seq[i:i+2]
                if phoneme_pair[0] in self.mora_rules and \
                   phoneme_pair[1] in self.mora_rules[phoneme_pair[0]]:
                    moras.append(self.mora_rules[phoneme_pair[0]][phoneme_pair[1]])
                    i += 2
                    continue
            
            # 単一音素の処理
            moras.append(phoneme_seq[i])
            i += 1
        
        return moras
    
    def compute_mora_error_rate(self, predictions: List[str], targets: List[str]) -> float:
        """モーラ単位のエラー率を計算"""
        total_distance = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # 音素列をモーラ列に変換
            pred_moras = self._convert_to_mora(pred)
            target_moras = self._convert_to_mora(target)
            
            # 編集距離の計算
            distance = editdistance.eval(pred_moras, target_moras)
            
            total_distance += distance
            total_length += len(target_moras)
        
        return total_distance / total_length if total_length > 0 else 1.0
    
    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """全ての評価指標を計算"""
        return {
            'phoneme_error_rate': self.compute_phoneme_error_rate(predictions, targets),
            'mora_error_rate': self.compute_mora_error_rate(predictions, targets)
        }