import torch
import torch.nn as nn
import copy

from VANP.utils.nn import get_output_size


class EndToEnd(nn.Module):

    def __init__(
        self,
        image_encoder: nn.Module = None,
        goal_encoder: nn.Module = None,
        controller: nn.Module = None,
        *,
        pred_len: int = 5,  # prediction length
        obs_len: int = 5,  # observation length
        action_size: int = 2,  # action space dim
        action_type: str = "xy",
        obs_context_size: int = 512,  # observation context size
        policy: str = "avg",
    ):
        super().__init__()
        assert policy.lower() in [
            "avg",
            "attn",
            "mlp",
        ], f"policy method can be either 'avg', 'mlp' or 'attn', but given {policy}!"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.action_size = action_size
        self.action_type = action_type
        self.obs_context_size = obs_context_size
        self.image_encoder = image_encoder
        self.goal_encoder = goal_encoder
        self.controller = controller
        img_encoder_output_size = get_output_size(
            copy.deepcopy(self.image_encoder), "conv", image_encoder.device
        )
        if img_encoder_output_size != self.obs_context_size:
            self.image_compressor = nn.Sequential(
                nn.Linear(img_encoder_output_size, self.obs_context_size),
                nn.LeakyReLU(),
            )
        else:
            self.image_compressor = nn.Identity()

    def forward(
        self,
        past_frames: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.obs_len == len(past_frames)
        bsz = goal.size(0)
        enc_goal = self.goal_encoder(goal.view(bsz, -1)).unsqueeze(1)
        imgs = torch.stack(
            [self.image_compressor(self.image_encoder(frame)) for frame in past_frames],
            dim=1,
        )  # (B, T, L)

        z = torch.cat([imgs, enc_goal], dim=1)  # concat along T dim
        actions, temp_distance = self.controller(z)
        actions = actions.view(bsz, self.pred_len, self.action_size)

        if self.action_type == "xy":
            actions[:, :, :2] = torch.cumsum(
                actions[:, :, :2], dim=1
            )  # convert deltas to positions

        return actions, temp_distance
