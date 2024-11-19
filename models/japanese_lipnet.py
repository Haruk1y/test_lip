import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conformer import ConformerEncoder
from models.frontend import ImageEncoder

class JapaneseLipNet(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        dropout=0.1,
        max_seq_length=459  # ROHANデータセットの最大フレーム数
    ):
        super().__init__()
        
        # 画像特徴抽出
        self.image_encoder = ImageEncoder(output_dim=d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Conformerエンコーダ
        self.conformer = ConformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # CTC用の出力層
        self.output_layer = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch, channel, time, height, width)
        batch_size = x.size(0)
        
        # 画像特徴抽出
        x = self.image_encoder(x)  # (batch, time, d_model)
        
        # Positional Encoding追加
        x = x + self.pos_encoder[:, :x.size(1)]
        
        # Conformerエンコーディング
        x = self.conformer(x)
        
        # CTC用にロジット変換
        logits = self.output_layer(x)  # (batch, time, num_classes)
        
        return logits
    
    def _create_padding_mask(self, seq_len, max_len):
        """系列長に基づくパディングマスクを生成"""
        mask = torch.ones(seq_len, max_len, dtype=torch.bool)
        for i in range(seq_len):
            mask[i, :min(i + 1, max_len)] = False
        return mask

def create_model(config):
    """設定からモデルを生成するヘルパー関数"""
    return JapaneseLipNet(
        num_classes=44,  # ROHANデータセットの音素数
        d_model=config.model.d_model,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
