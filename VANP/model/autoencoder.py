import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, action_type: str = "xy"):
        super().__init__()
        self.action_type = action_type
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L = x.shape
        z = self.encoder(x.view(B, -1))
        x_hat = self.decoder(z)
        x_hat = x_hat.view(B, T, L)
        if self.action_type == "xy":
            x_hat[:, :, :2] = torch.cumsum(
                x_hat[:, :, :2], dim=1
            )  # convert deltas to positions
            if L > 2:
                x_hat[:, :, 2:] = torch.nn.functional.normalize(
                    x_hat[:, :, 2:].clone(), dim=-1
                )  # normalize the angle prediction
        return z, x_hat


class VAE(nn.Module):
    def __init__(self, encoder, decoder, action_type: str = "xy"):
        super().__init__()
        self.action_type = action_type
        self.encoder = encoder
        self.decoder = decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        B, T, L = x.shape
        mean, log_var = self.encoder(x.view(B, -1))
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(B, T, L)
        if self.action_type == "xy":
            x_hat[:, :, :2] = torch.cumsum(
                x_hat[:, :, :2], dim=1
            )  # convert deltas to positions
            if L > 2:
                x_hat[:, :, 2:] = torch.nn.functional.normalize(
                    x_hat[:, :, 2:].clone(), dim=-1
                )  # normalize the angle prediction

        return x_hat, z, mean, log_var


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, n_action: int):
        super().__init__()
        self.linear = nn.Linear(n_action, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b, t, l = x.shape
        outputs = []  # to store embedding of each action
        # iterate over timestep and embed each action separately
        for i in range(t):
            outputs.append(self.ln(self.linear(x[:, i, :])))

        # stack actions together along the time dim -> (B, T, d_model)
        out = torch.stack(outputs, dim=1)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # we need 1 to broadcast along the time dimension
        self.positional_encoding = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        # calculate the positional encoding
        pe = self.positional_encoding[: x.shape[0]]
        return x + pe


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_action: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(d_hidden, n_action)

    def forward(self, x: torch.Tensor):
        # print("Inside regression head, before:", f"{x.shape = }")
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        # print("Inside regression head, after:", f"{x.shape = }")
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_head: int,
        n_action: int,
        d_hidden: int,
        pred_len: int,
        action_type: str = "xy",
        dropout: float = 0.5,
        n_registers: int = 4,
    ):
        super().__init__()
        self.action_type = action_type
        self.d_model = d_model
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_hidden, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, n_layers
        )
        self.patch_embedding = PatchEmbedding(d_model, n_action)
        self.positional_encoding = PositionalEncoding(d_model)
        self.regression_head = RegressionHead(d_model, d_hidden, pred_len * n_action)

        # cls token to save the context
        self.cls_token_emb = nn.Parameter(
            torch.randn(1, 1, d_model), requires_grad=True
        )
        # register tokens
        self.n_registers = n_registers
        self.reg_token_emb = nn.Parameter(
            torch.randn(1, self.n_registers, d_model), requires_grad=True
        )
        self.ln = nn.LayerNorm([d_model])

    def forward(self, x: torch.Tensor):
        B, T, L = x.shape
        # create patches
        x = self.patch_embedding(x)
        # create the cls token
        cls_token_emb = self.cls_token_emb.expand(B, -1, -1)
        reg_token_emb = self.reg_token_emb.expand(B, -1, -1)
        # print(f"{reg_token_emb.shape = }")
        # concat the cls token to the beginning of the sequence
        x = torch.cat([cls_token_emb, x, reg_token_emb], dim=1)
        # print(f"{x.shape = }")
        # add positional encodings
        x = self.positional_encoding(x)
        # pass to the transformer
        x = self.transformer_encoder(x)
        # print("After transformer", f"{x.shape = }")
        # retrieve cls token
        cls = x[:, 0, :]
        # print("CLS token", f"{x.shape = }")
        # pass through layer norm
        cls = self.ln(cls)
        # reconstruct actions
        x = self.regression_head(cls)
        x = x.view(B, T, L)
        # print(f"{x.shape = }")
        if self.action_type == "xy":
            x[:, :, :2] = torch.cumsum(
                x[:, :, :2], dim=1
            )  # convert deltas to positions
            if L > 2:
                x[:, :, 2:] = torch.nn.functional.normalize(
                    x[:, :, 2:].clone(), dim=-1
                )  # normalize the angle prediction
        return cls, x


if __name__ == "__main__":
    p = Transformer(4, 512, 8, 2, 256, 20, "xy", 0.3)
    b, t, l = 128, 20, 2
    input = torch.randn(b, t, l)
    print(f"{input.shape = }")
    o = p(input)
