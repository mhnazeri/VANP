import math

import torch
import torch.nn as nn


class TransformerPolicy(nn.Module):
    """Defines a transformer as the policy network"""

    def __init__(
        self,
        controller: nn.Module,
        d_model: int = 512,
        nhead: int = 4,
        d_hid: int = 512,
        num_layers: int = 4,
        dropout: float = 0.5,
        n_registers: int = 4,
    ) -> None:
        """
        Args:
            d_model: (int) the number of expected features in the input
            nhead: (int) the number of heads in the multiheadattention models
            d_hid: (int)
            num_layers: (int) the number of sub-encoder-layers in the encoder
            d_hid: (int) the dimensionality of the hidden network
            dropout: (float) dropout rate
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.controller = controller
        self.temp_predictor = nn.Linear(controller[0].in_features, 1)
        # ctx token to save the context
        self.ctx_token_emb = nn.Parameter(
            torch.randn(1, 1, d_model), requires_grad=True
        )
        # register tokens
        self.n_registers = n_registers
        if self.n_registers > 0:
            self.reg_token_emb = nn.Parameter(
                torch.randn(1, self.n_registers, d_model), requires_grad=True
            )
        self.ln = nn.LayerNorm([d_model])

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: (torch.Tensor) shape (B, T, L)
        """
        B, T, L = src.shape
        # create the ctx token
        ctx_token_emb = self.ctx_token_emb.expand(B, -1, -1)
        # concat the cls token to the beginning of the sequence
        tokens = [ctx_token_emb, src]

        if self.n_registers > 0:
            reg_token_emb = self.reg_token_emb.expand(B, -1, -1)
            tokens.append(reg_token_emb)
        src = torch.cat(tokens, dim=1)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        # retreive the cls/context token
        output = self.ln(output[:, 0, :])
        temp_distance = self.temp_predictor(output)
        actions = self.controller(output)
        return actions, temp_distance


class PoolingPolicy(nn.Module):
    """Defines a policy based on average pooling"""

    def __init__(self, controller: nn.Module, context_size: int):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(context_size)
        controller[0] = nn.Linear(context_size, controller[0].out_features)
        self.controller = controller
        self.temp_predictor = nn.Linear(context_size, 1)

    def forward(self, src: torch.Tensor):
        z = self.pooling(src.flatten(start_dim=1))
        actions = self.controller(z)
        temp_distance = self.temp_predictor(z)
        return actions, temp_distance


class MLPPolicy(nn.Module):
    def __init__(
        self,
        controller: nn.Module,
        obs_context_size: int,
        obs_len,
    ):
        super().__init__()
        controller[0] = nn.Linear(
            obs_context_size * (obs_len + 1), controller[0].out_features
        )
        self.controller = controller
        self.temp_predictor = nn.Linear(controller[0].in_features, 1)

    def forward(self, src: torch.Tensor):
        actions = self.controller(src)
        temp_distance = self.temp_predictor(src)
        return actions, temp_distance


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # broadcast over t dim.
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (torch.Tensor) shape [B, seq_len, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return x
