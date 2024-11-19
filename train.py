import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

from dataset import ROHANDataset
from models.japanese_lipnet import JapaneseLipNet
from utils.japanese_tokenizer import JapanesePhonemeTokenizer
from utils.metrics import compute_error_rate

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup tokenizer
        self.tokenizer = JapanesePhonemeTokenizer()
        
        # Create model
        self.model = JapaneseLipNet(
            num_classes=self.tokenizer.num_classes,
            d_model=config.d_model,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        ).to(self.device)
        
        if config.multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Cosine annealing scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(self.get_dataloader('train')),
            pct_start=0.1  # 10% warmup
        )
        
        # Loss function (CTC Loss)
        self.criterion = nn.CTCLoss(blank=self.tokenizer.blank_idx, zero_infinity=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Best metrics
        self.best_val_error = float('inf')
        
    def get_dataloader(self, split: str) -> DataLoader:
        """Create data loader for specified split"""
        dataset = ROHANDataset(
            video_dir=self.config.video_dir,
            lab_dir=self.config.lab_dir,
            landmark_dir=self.config.landmark_dir,
            file_list=f'{self.config.data_dir}/{split}_list.txt',
            tokenizer=self.tokenizer,
            vid_padding=self.config.vid_padding,
            txt_padding=self.config.txt_padding,
            phase=split
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        def validate_lengths(log_probs, targets, input_lengths, target_lengths):
            """入力とターゲットの長さを検証"""
            B, T, C = log_probs.size()
            max_target_length = targets.size(1)
            
            print(f"\nSequence length validation:")
            print(f"- Log probs shape: {log_probs.shape}")
            print(f"- Targets shape: {targets.shape}")
            print(f"- Input lengths: {input_lengths}")
            print(f"- Target lengths: {target_lengths}")
            
            if max_target_length < T:
                print(f"Warning: Target length ({max_target_length}) is less than input length ({T})")

        self.model.train()
        total_loss = 0
        total_error_phoneme = 0
        total_error_mora = 0
        num_batches = 0
        
        train_loader = self.get_dataloader('train')
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # データをGPUに転送
            videos = batch['vid'].to(self.device)
            texts = batch['txt'].to(self.device)
            vid_lengths = batch['vid_len'].to(self.device)
            txt_lengths = batch['txt_len'].to(self.device)
            
            # Forward pass
            logits = self.model(videos)
            
            # Loss計算
            log_probs = torch.log_softmax(logits, dim=-1)
            # Validate lengths on first batch
            if num_batches == 0:
                validate_lengths(log_probs, texts, vid_lengths, txt_lengths)
            loss = self.criterion(
                log_probs.transpose(0, 1),
                texts,
                vid_lengths,
                txt_lengths
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # メトリクスの計算
            with torch.no_grad():
                try:
                    # バッチ内の各サンプルに対してデコード
                    predictions = []
                    targets = []
                    
                    for i in range(logits.size(0)):
                        sample_logits = logits[i:i+1]  # (1, T, C)
                        pred = self.tokenizer.decode_ctc(sample_logits)
                        predictions.append(pred)
                        
                        target_length = txt_lengths[i].item()
                        target = texts[i, :target_length].cpu().numpy()
                        target = self.tokenizer.decode(target)
                        targets.append(target)
                    
                    # エラー率の計算
                    error_phoneme = compute_error_rate(predictions, targets, level='phoneme')
                    error_mora = compute_error_rate(predictions, targets, level='mora')
                    
                    # デバッグ情報（最初のバッチのみ）
                    if num_batches == 0:
                        print("\nFirst batch debug info:")
                        print(f"Predictions: {predictions[0][:10]}")
                        print(f"Targets: {targets[0][:10]}")
                    
                except Exception as e:
                    print(f"\nError in metrics calculation: {str(e)}")
                    print(f"Logits shape: {logits.shape}")
                    print(f"Logits device: {logits.device}")
                    raise
            
            total_loss += loss.item()
            total_error_phoneme += error_phoneme
            total_error_mora += error_mora
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ph_err': f'{error_phoneme:.4f}',
                'mora_err': f'{error_mora:.4f}'
            })
        
        # 平均メトリクスの計算
        avg_loss = total_loss / num_batches
        avg_error_phoneme = total_error_phoneme / num_batches
        avg_error_mora = total_error_mora / num_batches
        
        return avg_loss, avg_error_phoneme, avg_error_mora
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_error_phoneme = 0
        total_error_mora = 0
        num_batches = 0
        
        val_loader = self.get_dataloader('val')
        
        for batch in val_loader:
            videos = batch['vid'].to(self.device)
            texts = batch['txt'].to(self.device)
            vid_lengths = batch['vid_len'].to(self.device)
            txt_lengths = batch['txt_len'].to(self.device)
            
            # Forward pass
            logits = self.model(videos)
            
            # Compute loss
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = self.criterion(
                log_probs.transpose(0, 1),
                texts,
                vid_lengths,
                txt_lengths
            )
            
            # Compute metrics
            predictions = self.tokenizer.decode_ctc(logits)
            targets = [self.tokenizer.decode(txt[:length]) 
                      for txt, length in zip(texts.cpu().numpy(), txt_lengths.cpu().numpy())]
            
            error_phoneme = compute_error_rate(predictions, targets, level='phoneme')
            error_mora = compute_error_rate(predictions, targets, level='mora')
            
            total_loss += loss.item()
            total_error_phoneme += error_phoneme
            total_error_mora += error_mora
            num_batches += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_error_phoneme = total_error_phoneme / num_batches
        avg_error_mora = total_error_mora / num_batches
        
        # Log metrics
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/error_phoneme', avg_error_phoneme, epoch)
        self.writer.add_scalar('val/error_mora', avg_error_mora, epoch)
        
        # Save best model
        if avg_error_phoneme < self.best_val_error:
            self.best_val_error = avg_error_phoneme
            self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
        
        return avg_loss, avg_error_phoneme, avg_error_mora
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # Training phase
            train_loss, train_error_ph, train_error_mora = self.train_epoch(epoch)
            print(f"Training - Loss: {train_loss:.4f}, Phoneme Error: {train_error_ph:.4f}, Mora Error: {train_error_mora:.4f}")
            
            # Validation phase
            val_loss, val_error_ph, val_error_mora = self.validate(epoch)
            print(f"Validation - Loss: {val_loss:.4f}, Phoneme Error: {val_error_ph:.4f}, Mora Error: {val_error_mora:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_error': self.best_val_error,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--lab_dir', type=str, required=True)
    parser.add_argument('--landmark_dir', type=str, required=True)
    parser.add_argument('--vid_padding', type=int, default=200)
    parser.add_argument('--txt_padding', type=int, default=459)  # CTCの要件に合わせて増加xs
    
    # Other configuration
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--multi_gpu', action='store_true')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Start training
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()