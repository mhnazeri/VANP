import torch
from torch import nn
import copy

from VANP.utils.nn import get_output_size
from VANP.model.autoencoder import PositionalEncoding


class Projector(nn.Module):
    """Projector for Barlow Twins"""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PretextModel(nn.Module):
    def __init__(
        self,
        img_backbone,
        action_backbone,
        *,
        action_encoder_type: str = "mlp",
        feature_size: int = 512,
        projection_dim: int = 8192,
        hidden_dim: int = 8192,
        lamda: float = 0.005,
        corr_neg_one: bool = True,
        n_registers: int = 4,
        nhead: int = 4,
        num_layers: int = 4
    ):
        super().__init__()
        self.lamda = lamda
        self.action_encoder_type = action_encoder_type
        self.corr_neg_one = corr_neg_one
        self.projection_dim = projection_dim
        self.img_backbone = img_backbone
        self.action_backbone = action_backbone
        img_encoder_output_size = get_output_size(
            copy.deepcopy(self.img_backbone), "conv", img_backbone.device
        )
        if action_encoder_type == "mlp":
            action_encoder_output_size = self.action_backbone[-1].out_features
        elif action_encoder_type == "attn":
            action_encoder_output_size = self.action_backbone.d_model
        if img_encoder_output_size != feature_size:
            self.image_compressor = nn.Sequential(
                nn.Linear(img_encoder_output_size, feature_size), nn.LeakyReLU()
            )
        else:
            self.image_compressor = nn.Identity()
        if action_encoder_output_size != feature_size:
            self.action_compressor = nn.Sequential(
                nn.Linear(action_encoder_output_size, feature_size), nn.LeakyReLU()
            )
        else:
            self.action_compressor = nn.Identity()

        # project actions latent space
        self.projector = Projector(feature_size, hidden_dim, projection_dim)
        # Transformer module
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            feature_size, nhead, hidden_dim, 0.4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers
        )
        self.positional_encoding = PositionalEncoding(feature_size)

        # cls token to save the context
        self.ctx_token_emb = nn.Parameter(
            torch.randn(1, 1, feature_size), requires_grad=True
        )
        # register tokens
        self.n_registers = n_registers
        if n_registers > 0:
            self.reg_token_emb = nn.Parameter(
                torch.randn(1, n_registers, feature_size), requires_grad=True
            )

    def forward(self, frames, future_frame, actions):
        B, T, L = actions.size()
        future_frame = self.image_compressor(self.img_backbone(future_frame))
        # frames.append(future_frame) # if we want to consider goal as input
        frames = torch.stack(
            [self.image_compressor(self.img_backbone(frame)) for frame in frames], dim=1
        )  # (B, T, L)

        img_embed = frames[:, -1]
        ctx_token_emb = self.ctx_token_emb.expand(B, -1, -1)
        tokens = [ctx_token_emb, frames]
        if self.n_registers > 0:
            reg_token_emb = self.reg_token_emb.expand(B, -1, -1)
            tokens.append(reg_token_emb)
        # concat the ctx token to the beginning of the sequence
        x = torch.cat(tokens, dim=1)
        x = self.positional_encoding(x)
        # pass to the transformer
        x = self.transformer_encoder(x)[:, 0, :]
        img_z = self.projector(x)
        future_frame_z = self.projector(future_frame)
        if self.action_encoder_type == "mlp":
            action_embed = self.action_backbone(actions)
        # elif self.action_encoder_type == 'vae':
        #     _, action_embed, _, _ = self.action_backbone(actions)
        elif self.action_encoder_type == "attn":
            action_embed, _ = self.action_backbone(actions)
        action_z = self.projector(self.action_compressor(action_embed))

        return img_embed, img_z, action_z, future_frame_z


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == "__main__":
    pass
