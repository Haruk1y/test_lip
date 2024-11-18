import torch
import torch.nn as nn

class JapaneseLipNet(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        # Image Encoder
        self.image_encoder = ImageEncoder(output_dim=d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))  # Max sequence length: 1000
        
        # Conformer Encoder
        self.conformer = ConformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output Layer
        self.fc = nn.Linear(d_model, num_classes)
        
    def create_mask(self, src_len, tgt_len, device):
        # Create masks for transformer decoder
        src_mask = torch.zeros((src_len, src_len), device=device).bool()
        tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) * float('-inf'), diagonal=1).bool()
        
        return src_mask, tgt_mask
    
    def forward(self, x, target=None, target_mask=None):
        # x shape: (batch, channel, time, height, width)
        batch_size = x.size(0)
        
        # Extract visual features
        x = self.image_encoder(x)  # (batch, time, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1)]
        
        # Conformer encoding
        memory = self.conformer(x)
        
        if self.training and target is not None:
            # During training, use teacher forcing
            tgt = target[:, :-1]  # Remove last token
            tgt_mask = target_mask[:, :-1]
            
            # Create masks
            src_mask, tgt_mask = self.create_mask(memory.size(1), tgt.size(1), x.device)
            
            # Decode
            tgt = F.one_hot(tgt, num_classes=self.fc.out_features).float()
            tgt = torch.matmul(tgt, self.fc.weight)  # Project to d_model dimensions
            
            output = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
        else:
            # During inference, use autoregressive decoding
            output = self._autoregressive_decode(memory)
        
        # Project to vocabulary size
        logits = self.fc(output)
        
        return logits
    
    def _autoregressive_decode(self, memory, max_len=1000):
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with SOS token
        output = torch.zeros((batch_size, 1, self.fc.out_features), device=device)
        output[:, 0, 0] = 1  # SOS token
        
        for i in range(max_len - 1):
            # Project to d_model dimensions
            tgt = torch.matmul(output, self.fc.weight)
            
            # Create masks
            src_mask, tgt_mask = self.create_mask(memory.size(1), output.size(1), device)
            
            # Decode one step
            dec_output = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            
            # Project to vocabulary size and get prediction
            logits = self.fc(dec_output[:, -1:])
            pred = F.softmax(logits, dim=-1)
            
            # Concatenate prediction
            output = torch.cat([output, pred], dim=1)
            
            # Break if all sequences predict EOS token
            if (pred.argmax(dim=-1) == 1).all():  # Assuming 1 is EOS token
                break
        
        return output