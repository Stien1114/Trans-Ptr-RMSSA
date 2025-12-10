# -*- coding: utf-8 -*-
"""model.py

Using **Transformer Encoder + Pointer Network** to solve the **RMSSA** (Routing, Modulation, Spectrum and Space Assignment) problem in optical networks.

Improved version:
1. Enhanced positional encoding
2. Improved attention mechanism
3. Better feature fusion
4. Optimized initialization strategy
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helper Classes
# =============================================================================

class PositionalEncoding(nn.Module):
    """Improved sinusoidal positional encoding with learnable position bias."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Pre-generate positional encodings
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Learnable position bias
        self.position_bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [batch, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
        x = x + pos_emb + self.position_bias
        return self.dropout(x)


class RequestAwareAttention(nn.Module):
    """Simplified Multi-head Attention, optimized for RMSSA"""

    def __init__(self, d_model: int, static_size: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Standard Multi-head Attention
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Request feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Linear(static_size, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] Encoded features
            static: [batch, static_size, seq_len] Original static features (s,d,tr)
        """
        # Calculate request feature enhancement
        static_t = static.permute(0, 2, 1)  # [batch, seq_len, 3]
        feature_enhance = self.feature_enhance(static_t)  # [batch, seq_len, d_model]

        # Feature fusion
        enhanced_x = x + 0.1 * feature_enhance  # Slight enhancement to avoid destroying original features

        # Standard Multi-head Attention
        attn_out, _ = self.attention(enhanced_x, enhanced_x, enhanced_x)

        return attn_out


class TransformerEncoderLayer(nn.Module):
    """Simplified Transformer Encoder Layer"""

    def __init__(self, d_model: int,static_size: int, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = RequestAwareAttention(d_model, static_size, nhead, dropout)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        src2 = self.self_attn(src, static)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward with residual connection
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class RequestTransformerEncoder(nn.Module):
    """Transformer Encoder for communication requests (supports 11-dim features)"""

    def __init__(
            self,
            input_dim: int,  # Can now be 3 or 11
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 3,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim

        if input_dim == 11:
            # Group processing for different types of features
            self.basic_proj = nn.Linear(3, d_model // 2)
            self.path_proj = nn.Linear(5, d_model // 4)
            self.fs_proj = nn.Linear(3, d_model // 4)

            self.feature_fusion = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            # Original 3-dim features
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model,
                                    static_size=self.input_dim,  # Added
                                    nhead=nhead,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, src: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [batch, seq_len, input_dim] Input features
            static: [batch, input_dim, seq_len] Static features (for compatibility)
        """
        if self.input_dim == 11:
            # Group processing for 11-dim features
            basic_features = src[:, :, :3]
            path_features = src[:, :, 3:8]
            fs_features = src[:, :, 8:11]

            basic_emb = self.basic_proj(basic_features)
            path_emb = self.path_proj(path_features)
            fs_emb = self.fs_proj(fs_features)

            combined = torch.cat([basic_emb, path_emb, fs_emb], dim=-1)
            x = self.feature_fusion(combined)
        else:
            # Original 3-dim feature processing
            x = self.input_proj(src)

        # Positional encoding
        x = self.pos_enc(x)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, static)

        # Output projection
        x = self.output_proj(x)
        return x


class PointerDecoder(nn.Module):
    """Improved Pointer Network Decoder"""

    def __init__(self, d_model: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # LSTM used to maintain decoding state
        self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True)

        # Improved Attention Mechanism - Using Bahdanau attention
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_k = nn.Linear(d_model, hidden_size, bias=True)
        self.W_v = nn.Linear(d_model, hidden_size, bias=False)

        # Using single linear layer instead of parameter vector, more stable
        self.attention_combine = nn.Linear(hidden_size, 1, bias=False)

        # Optional value projection
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Weight initialization"""
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, enc_outputs: torch.Tensor, mask: torch.Tensor,
                last_selected: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.distributions.Categorical, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Using Additive Attention (Bahdanau attention) to avoid gradient saturation
        """
        batch_size, seq_len, _ = enc_outputs.shape
        device = enc_outputs.device

        # Prepare LSTM input
        if last_selected is None:
            # Use learned start token or mean embedding
            lstm_input = enc_outputs.mean(dim=1).unsqueeze(1)
        else:
            # Use embedding of previously selected item
            indices = last_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.d_model)
            lstm_input = enc_outputs.gather(1, indices)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        query = self.W_q(lstm_out.squeeze(1))  # [batch, hidden_size]

        # Calculate keys and values
        keys = self.W_k(enc_outputs)  # [batch, seq_len, hidden_size]
        values = self.W_v(enc_outputs)  # [batch, seq_len, hidden_size]

        # Additive Attention (Bahdanau attention) - Avoid double tanh
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_size]

        # Use additive attention instead of double tanh
        attention_input = keys + query_expanded  # [batch, seq_len, hidden_size]
        attention_input = self.layer_norm(attention_input)  # Normalize to stabilize training

        # Calculate attention scores - Using only one tanh
        scores = self.attention_combine(torch.tanh(attention_input)).squeeze(-1)  # [batch, seq_len]

        # Apply mask - Fix NaN issue
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Check if all values are masked
        # If all values in a batch are -inf, we need special handling
        all_masked = (mask.sum(dim=1) == 0)

        if all_masked.any():
            # Create uniform distribution for fully masked batches
            # First calculate normal softmax
            attention_weights = F.softmax(scores, dim=-1)

            # For fully masked batches, set to uniform distribution
            uniform_probs = torch.ones_like(scores) / seq_len
            attention_weights = torch.where(
                all_masked.unsqueeze(1),
                uniform_probs,
                attention_weights
            )
        else:
            # Calculate attention weights normally
            attention_weights = F.softmax(scores, dim=-1)

        # Ensure no NaN
        attention_weights = torch.nan_to_num(attention_weights, nan=1.0 / seq_len)

        # Ensure probabilities sum to 1 (numerical stability)
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)

        # Optional: Aggregate values using attention weights (for richer context)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch, hidden_size]

        return torch.distributions.Categorical(probs=attention_weights), hidden


class HistoryAwarePointerDecoder(nn.Module):
    """
    Pointer Decoder with full history memory
    Records all assigned requests in the current episode
    """

    def __init__(self, d_model: int, hidden_size: int = 128,
                 max_seq_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len  # Max sequence length

        # Use LSTM to handle sequence information (including current and history)
        self.lstm = nn.LSTM(d_model, hidden_size,
                            batch_first=True, num_layers=2,
                            dropout=dropout if dropout > 0 else 0)

        # History Encoder - Process all assigned requests
        self.history_encoder = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-head Self-Attention for integrating historical information
        self.history_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=dropout, batch_first=True
        )

        # History Aggregation method
        self.history_aggregation = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # concat current query and history
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Pointer Attention Mechanism
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(d_model, hidden_size)
        self.W_history = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, 1)

        # Layer Normalization
        self.layer_norm_q = nn.LayerNorm(hidden_size)
        self.layer_norm_k = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, enc_outputs: torch.Tensor, mask: torch.Tensor,
                last_selected: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                history_embeddings: Optional[torch.Tensor] = None,
                step: int = 0):
        """
        Args:
            enc_outputs: [batch, seq_len, d_model] Encodings of all candidate requests
            mask: [batch, seq_len] Mask for available positions
            last_selected: [batch] Index of last selection
            hidden: LSTM hidden state
            history_embeddings: [batch, step, d_model] Embeddings of all selected requests
            step: Current step number (how many requests have been assigned)
        """
        batch_size, seq_len, _ = enc_outputs.shape
        device = enc_outputs.device

        # 1. Get current input embedding
        if last_selected is None:
            # First step: use mean of all candidates as initial input
            current_embed = enc_outputs.mean(dim=1, keepdim=True)
            # Initialize history
            if history_embeddings is None:
                history_embeddings = torch.zeros(batch_size, 0, self.d_model, device=device)
        else:
            # Get embedding of last selection
            indices = last_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.d_model)
            current_embed = enc_outputs.gather(1, indices)

            # Add to history
            if history_embeddings is None:
                history_embeddings = current_embed
            else:
                # Accumulate all selected requests
                history_embeddings = torch.cat([history_embeddings, current_embed], dim=1)

        # 2. LSTM processes current input
        lstm_out, hidden = self.lstm(current_embed, hidden)
        query_base = lstm_out.squeeze(1)  # [batch, hidden_size]

        # 3. Process history information (if available)
        if history_embeddings is not None and history_embeddings.size(1) > 0:
            # Encode history embeddings
            history_encoded = self.history_encoder(history_embeddings)  # [batch, num_history, hidden_size]

            # Use self-attention to integrate history information
            history_context, _ = self.history_attention(
                history_encoded, history_encoded, history_encoded
            )  # [batch, num_history, hidden_size]

            # Aggregate history information (can use mean, max, or weighted sum)
            # Here use weighted average, weights decay over time
            num_history = history_context.size(1)
            if num_history > 0:
                # Create time decay weights (recent weights are larger)
                time_weights = torch.arange(1, num_history + 1, device=device, dtype=torch.float32)
                time_weights = F.softmax(time_weights / 2.0, dim=0)  # temperature=2.0
                time_weights = time_weights.unsqueeze(0).unsqueeze(-1)  # [1, num_history, 1]

                # Weighted aggregation
                history_summary = (history_context * time_weights).sum(dim=1)  # [batch, hidden_size]
            else:
                history_summary = torch.zeros(batch_size, self.hidden_size, device=device)

            # Combine current query and history information
            query = self.history_aggregation(
                torch.cat([query_base, history_summary], dim=-1)
            )  # [batch, hidden_size]
        else:
            query = query_base

        # 4. Calculate pointer attention scores
        query = self.layer_norm_q(self.W_q(query))  # [batch, hidden_size]
        keys = self.layer_norm_k(self.W_k(enc_outputs))  # [batch, seq_len, hidden_size]

        # Expand query to match dimensions of keys
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_size]

        # Additive Attention
        attention_input = torch.tanh(query_expanded + keys)  # [batch, seq_len, hidden_size]

        # Dropout
        if self.training:
            attention_input = self.dropout(attention_input)

        # Calculate scores
        scores = self.W_v(attention_input).squeeze(-1)  # [batch, seq_len]

        # 5. Apply mask
        scores_masked = scores.masked_fill(mask == 0, -1e9)

        # Handle fully masked cases
        valid_positions = mask.sum(dim=1)
        all_masked = (valid_positions == 0)
        if all_masked.any():
            scores_masked = torch.where(
                all_masked.unsqueeze(1) & (torch.arange(seq_len, device=device) == 0),
                torch.zeros_like(scores_masked),
                scores_masked
            )

        return torch.distributions.Categorical(logits=scores_masked), hidden, history_embeddings

class RobustPointerDecoder(nn.Module):
    """More robust Pointer Network Decoder, handling numerical stability"""

    def __init__(self, d_model: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # LSTM used to maintain decoding state
        self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True)

        # Attention Mechanism
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(d_model, hidden_size)

        # Use Bilinear Attention for better stability
        self.attention = nn.Bilinear(hidden_size, hidden_size, 1)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(0.1)
        self.layer_norm_q = nn.LayerNorm(hidden_size)
        self.layer_norm_k = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain to improve stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, enc_outputs: torch.Tensor, mask: torch.Tensor,
                last_selected: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.distributions.Categorical, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, seq_len, _ = enc_outputs.shape
        device = enc_outputs.device

        # Prepare LSTM input
        if last_selected is None:
            lstm_input = enc_outputs.mean(dim=1).unsqueeze(1)
        else:
            indices = last_selected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.d_model)
            lstm_input = enc_outputs.gather(1, indices)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        query = lstm_out.squeeze(1)  # [batch, hidden_size]

        # Projection and Normalization
        query = self.layer_norm_q(self.W_q(query))  # [batch, hidden_size]
        keys = self.layer_norm_k(self.W_k(enc_outputs))  # [batch, seq_len, hidden_size]

        # Calculate attention scores
        # Use Bilinear Attention, more stable
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_size]
        scores = self.attention(query_expanded, keys).squeeze(-1)  # [batch, seq_len]

        # Dropout for regularization
        if self.training:
            scores = self.dropout(scores)

        # Create a very large negative number, but not -inf (to avoid numerical issues)
        MASKING_VALUE = -1e9

        # Apply mask
        scores_masked = scores.masked_fill(mask == 0, MASKING_VALUE)

        # Check if any batch is fully masked
        valid_positions = mask.sum(dim=1)  # [batch]
        all_masked = (valid_positions == 0)

        # For fully masked batches, create a dummy valid position
        if all_masked.any():
            # Set first position as valid (avoid NaN)
            scores_masked = torch.where(
                all_masked.unsqueeze(1) & (torch.arange(seq_len, device=device) == 0),
                torch.zeros_like(scores_masked),  # Set first position to 0
                scores_masked
            )

        # Create distribution using logits (more stable)
        return torch.distributions.Categorical(logits=scores_masked), hidden


class DRL4RMSSA(nn.Module):
    """Improved DRL model - correctly handles 11-dimensional features"""

    def __init__(
            self,
            static_size: int,
            hidden_size: int,
            num_nodes: int,
            update_fn: Optional[Callable] = None,
            mask_fn: Optional[Callable] = None,
            transformer_layers: int = 3,
            transformer_heads: int = 8,
            dropout: float = 0.1,
            use_robust_decoder: bool = True,
            use_history: bool = True,
    ) -> None:
        super().__init__()
        self.static_size = static_size
        self.d_model = hidden_size
        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.use_history = use_history
        self.num_nodes = num_nodes

        # Ensure hidden_size is divisible by heads
        assert hidden_size % transformer_heads == 0

        if static_size == 11:
            # === Improved 11-dimensional feature processing ===
            h = hidden_size
            self.node_embed_dim = h // 4  # 64 (if h=256)

            # Node ID embedding
            self.src_embed = nn.Embedding(num_nodes, self.node_embed_dim)
            self.dst_embed = nn.Embedding(num_nodes, self.node_embed_dim)

            # Traffic feature processing (handled separately because it is important)
            self.traffic_proj = nn.Sequential(
                nn.Linear(1, h // 8),
                nn.ReLU(),
                nn.Linear(h // 8, h // 8),
                nn.LayerNorm(h // 8)
            )

            # Path feature processing (5 dims: avg_length, min_length, avg_hops, min_hops, avg_modulation)
            self.path_proj = nn.Sequential(
                nn.Linear(5, h // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h // 4, h // 4),
                nn.LayerNorm(h // 4)
            )

            # FS feature processing (3 dims: fs_min, fs_avg, fs_max)
            self.fs_proj = nn.Sequential(
                nn.Linear(3, h // 8),
                nn.ReLU(),
                nn.Linear(h // 8, h // 8),
                nn.LayerNorm(h // 8)
            )

            # Feature fusion
            concat_dim = self.node_embed_dim * 2 + h // 8 + h // 4 + h // 8  # Sum should equal h
            self.feature_fusion = nn.Sequential(
                nn.Linear(concat_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # Add an extra projection layer to enhance expressiveness
            self.enhancement_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        else:
            # Original processing for 3-dimensional features
            self.input_proj = nn.Sequential(
                nn.Linear(static_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_size, dropout)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=transformer_layers
        )

        # Pointer Decoder
        if use_history:
            self.pointer = HistoryAwarePointerDecoder(
                hidden_size, hidden_size, max_seq_len=200, dropout=dropout
            )
        elif use_robust_decoder:
            self.pointer = RobustPointerDecoder(hidden_size, hidden_size)
        else:
            self.pointer = PointerDecoder(hidden_size, hidden_size)

        self._init_parameters()

    def _build_encoder_input(self, static: torch.Tensor) -> torch.Tensor:
        """
        Improved feature processing
        static : [B, 11, S]
        Returns: [B, S, hidden_size]
        """
        B, _, S = static.shape
        device = static.device

        # 1. Extract Node IDs (keep as integers)
        s_idx = static[:, 0, :].long()  # [B, S]
        d_idx = static[:, 1, :].long()  # [B, S]

        # Node embeddings
        src_emb = self.src_embed(s_idx)  # [B, S, node_embed_dim]
        dst_emb = self.dst_embed(d_idx)  # [B, S, node_embed_dim]

        # 2. Traffic features (already normalized)
        traffic = static[:, 2:3, :].permute(0, 2, 1)  # [B, S, 1]
        traffic_norm = (traffic - 100) / 900  # Normalize to [0,1]
        traffic_feat = self.traffic_proj(traffic_norm)  # [B, S, h/8]

        # 3. Path features (indices 3-7, already normalized)
        path_features = static[:, 3:8, :].permute(0, 2, 1)  # [B, S, 5]
        path_feat = self.path_proj(path_features)  # [B, S, h/4]

        # 4. FS features (indices 8-10, already normalized)
        fs_features = static[:, 8:11, :].permute(0, 2, 1)  # [B, S, 3]
        fs_feat = self.fs_proj(fs_features)  # [B, S, h/8]

        # 5. Concatenate all features
        combined = torch.cat([
            src_emb,  # node_embed_dim
            dst_emb,  # node_embed_dim
            traffic_feat,  # h/8
            path_feat,  # h/4
            fs_feat  # h/8
        ], dim=-1)

        # 6. Feature fusion
        fused = self.feature_fusion(combined)  # [B, S, hidden_size]

        # 7. Enhancement layer
        enhanced = self.enhancement_layer(fused)  # [B, S, hidden_size]

        return enhanced

    def _init_parameters(self):
        """Improved parameter initialization"""
        for name, param in self.named_parameters():
            if 'embed' in name:
                # Initialize embeddings with normal distribution
                nn.init.normal_(param, mean=0, std=0.02)
            elif 'weight' in name and param.dim() > 1:
                # Initialize other weights with Xavier
                nn.init.xavier_uniform_(param, gain=0.8)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, static: torch.Tensor, x0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        device = static.device
        batch_size, feature_dim, seq_len = static.size()

        # Build encoder input
        if self.static_size == 11:
            enc_input = self._build_encoder_input(static)
        else:
            # Processing for 3-dim features
            static_t = static.permute(0, 2, 1)
            enc_input = self.input_proj(static_t)

        # Positional encoding
        enc_input = self.pos_enc(enc_input)

        # Transformer encoding
        enc_output = self.encoder(enc_input)

        # Pointer decoding
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        selections: List[torch.Tensor] = []
        logps: List[torch.Tensor] = []

        hidden = None
        last_selected = None
        history_embeddings = None

        for step in range(seq_len):
            if self.use_history:
                dist, hidden, history_embeddings = self.pointer(
                    enc_output, mask, last_selected, hidden, history_embeddings, step
                )
            else:
                dist, hidden = self.pointer(enc_output, mask, last_selected, hidden)

            # Sample
            idx = dist.sample()
            selections.append(idx)
            logps.append(dist.log_prob(idx))

            # Update state
            last_selected = idx

            # Update mask
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, idx)
            else:
                mask = mask.scatter(1, idx.unsqueeze(1), False)

        return torch.stack(selections, dim=1), torch.stack(logps, dim=1)

    @torch.no_grad()
    def greedy(self, static: torch.Tensor) -> torch.Tensor:
        """Greedy decoding"""
        self.eval()
        device = static.device
        batch_size, _, seq_len = static.size()

        # Build encoder input
        if self.static_size == 11:
            enc_input = self._build_encoder_input(static)
        else:
            static_t = static.permute(0, 2, 1)
            enc_input = self.input_proj(static_t)

        enc_input = self.pos_enc(enc_input)
        enc_output = self.encoder(enc_input)

        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        selections = []
        hidden = None
        last_selected = None
        history_embeddings = None

        for step in range(seq_len):
            if self.use_history:
                dist, hidden, history_embeddings = self.pointer(
                    enc_output, mask, last_selected, hidden, history_embeddings, step
                )
            else:
                dist, hidden = self.pointer(enc_output, mask, last_selected, hidden)

            idx = dist.probs.argmax(dim=-1)
            selections.append(idx)
            last_selected = idx

            if self.mask_fn is not None:
                mask = self.mask_fn(mask, idx)
            else:
                mask = mask.scatter(1, idx.unsqueeze(1), False)

        return torch.stack(selections, dim=1)
# =============================================================================
# Compatible Aliases
# =============================================================================

class Encoder(nn.Module):
    """Encoder interface compatible with old versions"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Simplified 1D convolution encoder, used for critic
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, input_size, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x