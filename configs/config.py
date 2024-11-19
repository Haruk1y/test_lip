from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class ModelConfig:
    # Model Architecture
    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # CNN Frontend
    cnn_dropout: float = 0.2
    
    # Image settings
    img_height: int = 64
    img_width: int = 128
    
    # Sequence lengths
    max_vid_length: int = 200
    max_txt_length: int = 200

@dataclass
class TrainingConfig:
    # 基本設定
    batch_size: int = 8  # GPUメモリに応じて調整
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0
    
    # オプティマイザー設定
    warmup_steps: int = 1000
    
    # デバイス設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu: bool = False
    
    # データローディング
    num_workers: int = 4  # CPU数に応じて調整
    pin_memory: bool = True
    
    # モデル設定
    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # シーケンス長設定
    max_vid_length: int = 459  # ROHANの最大フレーム数
    max_txt_length: int = 200

@dataclass
class DataConfig:
    # Data paths
    data_dir: Path
    video_dir: Path
    lab_dir: Path
    landmark_dir: Path
    
    # Data split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Data processing
    fps: float = 29.95
    use_landmarks: bool = True

@dataclass
class Config:
    # Top level config that combines all others
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig(
        data_dir=Path("data"),
        video_dir=Path("data/videos"),
        lab_dir=Path("data/lab"),
        landmark_dir=Path("data/landmarks")
    )
    
    # Paths for saving and logging
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    
    # Create all necessary directories
    def create_directories(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        return cls(
            model=ModelConfig(
                d_model=args.d_model,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                num_heads=args.num_heads,
                dropout=args.dropout
            ),
            training=TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                multi_gpu=args.multi_gpu
            ),
            data=DataConfig(
                data_dir=Path(args.data_dir),
                video_dir=Path(args.video_dir),
                lab_dir=Path(args.lab_dir),
                landmark_dir=Path(args.landmark_dir)
            )
        )