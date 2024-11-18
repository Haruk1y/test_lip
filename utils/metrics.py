from typing import List, Tuple
import numpy as np
import editdistance
from collections import defaultdict

class MetricsCalculator:
    """日本語読唇システムの評価指標を計算するクラス"""
    
    @staticmethod
    def compute_phoneme_error_rate(predictions: List[str], targets: List[str]) -> float:
        """
        音素単位のエラー率を計算
        
        Args:
            predictions: 予測された音素列のリスト
            targets: 正解の音素列のリスト
            
        Returns:
            float: エラー率 (0.0 ~ 1.0)
        """
        total_distance = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # 文字列を音素リストに変換
            pred_phones = list(pred)
            target_phones = list(target)
            
            # 編集距離の計算
            distance = editdistance.eval(pred_phones, target_phones)
            
            total_distance += distance
            total_length += len(target_phones)
        
        if total_length == 0:
            return 1.0
            
        return total_distance / total_length

    @staticmethod
    def compute_mora_error_rate(predictions: List[str], targets: List[str]) -> float:
        """
        モーラ単位のエラー率を計算
        
        Args:
            predictions: 予測された音素列のリスト
            targets: 正解の音素列のリスト
            
        Returns:
            float: エラー率 (0.0 ~ 1.0)
        """
        total_distance = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # 音素列をモーラ列に変換
            pred_moras = MetricsCalculator._convert_to_moras(pred)
            target_moras = MetricsCalculator._convert_to_moras(target)
            
            # 編集距離の計算
            distance = editdistance.eval(pred_moras, target_moras)
            
            total_distance += distance
            total_length += len(target_moras)
        
        if total_length == 0:
            return 1.0
            
        return total_distance / total_length

    @staticmethod
    def _convert_to_moras(phoneme_seq: str) -> List[str]:
        """
        音素列をモーラ列に変換
        
        Args:
            phoneme_seq: 音素列
            
        Returns:
            List[str]: モーラ列
        """
        # 特殊トークンの処理
        special_tokens = {'sil', 'sp', 'silB', 'silE', 'pau'}
        if phoneme_seq in special_tokens:
            return [phoneme_seq]
            
        # 音素の組み合わせルール
        mora_rules = {
            # 長音
            'a': {'a': 'aa'},
            'i': {'i': 'ii'},
            'u': {'u': 'uu'},
            'e': {'e': 'ee'},
            'o': {'o': 'oo'},
            
            # 撥音
            'N': {'': 'N'},
            
            # 促音
            'q': {'': 'q'},
            
            # 拗音
            'y': {'a': 'ya', 'u': 'yu', 'o': 'yo'},
            
            # 特殊な音素の組み合わせ
            'ch': {'a': 'cha', 'i': 'chi', 'u': 'chu', 'e': 'che', 'o': 'cho'},
            'sh': {'a': 'sha', 'i': 'shi', 'u': 'shu', 'e': 'she', 'o': 'sho'},
            'ts': {'u': 'tsu'},
        }
        
        moras = []
        i = 0
        while i < len(phoneme_seq):
            current_mora = ""
            
            # 2文字の特殊な組み合わせをチェック
            if i + 1 < len(phoneme_seq) and phoneme_seq[i:i+2] in {'ch', 'sh', 'ts'}:
                if i + 2 < len(phoneme_seq) and phoneme_seq[i+2] in mora_rules[phoneme_seq[i:i+2]]:
                    current_mora = mora_rules[phoneme_seq[i:i+2]][phoneme_seq[i+2]]
                    i += 3
                else:
                    current_mora = phoneme_seq[i]
                    i += 1
            # 拗音をチェック
            elif i + 1 < len(phoneme_seq) and phoneme_seq[i] in mora_rules and \
                 phoneme_seq[i+1] in mora_rules[phoneme_seq[i]]:
                current_mora = mora_rules[phoneme_seq[i]][phoneme_seq[i+1]]
                i += 2
            # 単一の音素
            else:
                current_mora = phoneme_seq[i]
                i += 1
            
            moras.append(current_mora)
        
        return moras

    @staticmethod
    def compute_detailed_metrics(predictions: List[str], targets: List[str]) -> dict:
        """
        詳細な評価指標を計算
        
        Args:
            predictions: 予測された音素列のリスト
            targets: 正解の音素列のリスト
            
        Returns:
            dict: 各種評価指標を含む辞書
        """
        metrics = {
            'phoneme_error_rate': MetricsCalculator.compute_phoneme_error_rate(predictions, targets),
            'mora_error_rate': MetricsCalculator.compute_mora_error_rate(predictions, targets),
        }
        
        # 音素ごとの混同行列の計算
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for pred, target in zip(predictions, targets):
            pred_phones = list(pred)
            target_phones = list(target)
            
            # 音素アライメント
            alignment = MetricsCalculator._align_sequences(pred_phones, target_phones)
            
            for p, t in alignment:
                confusion_matrix[t][p] += 1
        
        metrics['confusion_matrix'] = dict(confusion_matrix)
        
        return metrics

    @staticmethod
    def _align_sequences(seq1: List[str], seq2: List[str]) -> List[Tuple[str, str]]:
        """
        2つの配列を動的計画法でアライメント
        
        Args:
            seq1: 1つ目の配列
            seq2: 2つ目の配列
            
        Returns:
            List[Tuple[str, str]]: アライメント結果
        """
        # 動的計画法による編集距離の計算
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)
        
        # DPテーブルの初期化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # DPテーブルの計算
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        
        # バックトラックによるアライメントの取得
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq1[i-1] == seq2[j-1]:
                alignment.append((seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
                alignment.append((seq1[i-1], '*'))  # 削除
                i -= 1
            else:
                alignment.append(('*', seq2[j-1]))  # 挿入
                j -= 1
        
        return list(reversed(alignment))

    @staticmethod
    def format_metrics_report(metrics: dict) -> str:
        """
        評価指標を読みやすい形式でフォーマット
        
        Args:
            metrics: 評価指標を含む辞書
            
        Returns:
            str: フォーマットされた評価レポート
        """
        report = []
        report.append("=== 評価結果 ===")
        report.append(f"音素エラー率: {metrics['phoneme_error_rate']:.4f}")
        report.append(f"モーラエラー率: {metrics['mora_error_rate']:.4f}")
        
        if 'confusion_matrix' in metrics:
            report.append("\n=== 混同行列 ===")
            confusion_matrix = metrics['confusion_matrix']
            phonemes = sorted(set(sum([list(d.keys()) for d in confusion_matrix.values()], [])))
            
            # ヘッダーの作成
            header = "予測→\n正解↓"
            for p in phonemes:
                header += f"\t{p}"
            report.append(header)
            
            # 各行の作成
            for t in sorted(confusion_matrix.keys()):
                row = f"{t}"
                for p in phonemes:
                    row += f"\t{confusion_matrix[t].get(p, 0)}"
                report.append(row)
        
        return "\n".join(report)