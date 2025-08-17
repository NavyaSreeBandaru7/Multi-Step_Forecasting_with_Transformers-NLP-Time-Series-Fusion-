# File: src/tft_model.py
"""
Advanced Temporal Fusion Transformer implementation with:
- Multi-head attention
- Variable selection networks
- Interpretable components
- Quantile loss for uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class TimeDistributed(nn.Module):
    """Apply module across time dimension"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(t * n, -1)
        y = self.module(x_reshaped)
        return y.view(t, n, -1)

class GatedResidualNetwork(nn.Module):
    """Advanced feature processing with gating"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.gate = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
            
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(self.gate(x), dim=-1) * skip
        return self.layer_norm(x)

class VariableSelectionNetwork(nn.Module):
    """Feature selection for static and temporal features"""
    def __init__(self, input_size, num_vars, hidden_size, dropout=0.1):
        super().__init__()
        self.grns = nn.ModuleList()
        for i in range(num_vars):
            self.grns.append(
                GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
        self.grn_concat = GatedResidualNetwork(
            input_size=num_vars * hidden_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            dropout=dropout
        )

    def forward(self, embedding, variables):
        # Process each variable independently
        var_outputs = []
        for i in range(variables.size(-1)):
            var = variables[..., i].unsqueeze(-1)
            var_outputs.append(self.grns[i](var))
            
        # Concatenate and process
        concat = torch.cat(var_outputs, dim=-1)
        weights = self.grn_concat(embedding)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)
        
        # Weighted sum
        return torch.sum(concat * weights, dim=-2)

class TemporalFusionTransformer(nn.Module):
    """State-of-the-art temporal fusion transformer"""
    
    def __init__(self, dataset, hidden_size=64, lstm_layers=2, 
                 attention_head_size=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        
        # Static context
        self.static_context = nn.Sequential(
            nn.Embedding(1000, hidden_size),  # Series ID embedding
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Encoder/Decoder processing
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Temporal processing
        self.temporal_vsn = VariableSelectionNetwork(
            input_size=hidden_size,
            num_vars=2,  # time + value
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Attention mechanism
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_head_size,
            dropout=dropout,
            batch_first=True
        )
        
        # Forecasting output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # 3 quantiles: 10%, 50%, 90%
        )
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract inputs
        encoder_cont = x['encoder_cont']
        decoder_cont = x['decoder_cont']
        encoder_cat = x['encoder_cat']
        decoder_cat = x['decoder_cat']
        
        # Static context
        static_embed = self.static_context(encoder_cat[:, 0])
        
        # Temporal processing
        encoder_temporal = self.temporal_vsn(static_embed, encoder_cont)
        decoder_temporal = self.temporal_vsn(static_embed, decoder_cont)
        
        # LSTM processing
        encoder_out, (h, c) = self.encoder_lstm(encoder_temporal)
        decoder_out, _ = self.decoder_lstm(decoder_temporal, (h, c))
        
        # Attention across time
        attn_out, _ = self.multihead_attn(
            query=decoder_out,
            key=encoder_out,
            value=encoder_out
        )
        
        # Combine with residual
        combined = decoder_out + attn_out
        
        # Quantile predictions
        predictions = self.fc(combined)
        return predictions
    
    def predict(self, dataloader, device='cuda'):
        """Production-ready prediction method"""
        self.eval()
        self.to(device)
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self(batch)
                predictions.append(out.cpu())
                
        return torch.cat(predictions, dim=0)
