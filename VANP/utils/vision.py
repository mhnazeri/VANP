"""Utility functions related to vision"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import cv2
import pickle
from scipy.spatial.transform import Rotation as R


def plot_images(batch: torch.Tensor, title: str):
    """Plot a batch of images

    Args:
        batch: (torch.Tensor) a batch of images with dimensions (batch, channels, height, width)
        title: (str) title of the plot and saved file
    """
    n_samples = batch.size(0)
    plt.figure(figsize=(n_samples // 2, n_samples // 2))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0))
    )
    plt.savefig(f"{title}.png")


def draw_on_image(img, true_actions, action, gt=True):
    """Draw text on the image

    Args:
        img: (torch.Tensor) frame
        measurements: (dict) ground truth actions
        action: (torch.Tensor) predicted actions
        gt: whether to draw true action or not
    """
    # if measurements:
    linear_gt = true_actions[0].item()
    angular_gt = true_actions[1].item()

    linear = action[0].item()
    angular = action[1].item()

    upsampler = torch.nn.Upsample(scale_factor=2, mode="bilinear")
    if len(img.shape) < 4:
        img.unsqueeze_(0)
    img = upsampler(img).squeeze(0).permute(1, 2, 0).numpy()
    img_width = img.shape[1] // 2
    img = Image.fromarray(
        (((img - img.min()) / (-img.min() + img.max())) * 255).astype(np.uint8)
    )
    draw = ImageDraw.Draw(img)
    # load font
    fnt_path = Path(__file__).parent.parent / "misc_files/FUTURAM.ttf"
    fnt = ImageFont.truetype(str(fnt_path), 18)
    draw.text((5, 30), f"Linear_Vel: {linear:.3f}", fill="red", font=fnt)
    draw.text((5, 60), f"Angular_Vel: {angular:.3f}", fill="red", font=fnt)

    if gt:
        draw.text(
            (img_width * 2 - 200, 30),
            f"Linear_Vel: {linear_gt:.3f}",
            fill="green",
            font=fnt,
        )
        draw.text(
            (img_width * 2 - 200, 60),
            f"Angular_Vel: {angular_gt:.3f}",
            fill="green",
            font=fnt,
        )

    return np.array(img)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[-traj_len:, 0], 2, full=True)[1]
    res_y = np.polyfit(t, traj[-traj_len:, 1], 2, full=True)[1]
    print(f"{res_x + res_y = }")
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def to_robot(p1_batch, p2_batch):

    # Ensure input dimensions are correct
    if p1_batch.shape != p2_batch.shape or p1_batch.shape[1] != 6:
        raise ValueError(
            "Input batches must be of the same shape and contain 6 elements per pose"
        )

    n = p1_batch.shape[0]
    transformations = np.zeros((n, 6))

    # Extract and reshape translation components
    t1 = p1_batch[:, :3].reshape(n, 3, 1)
    t2 = p2_batch[:, :3].reshape(n, 3, 1)
    R1 = R.from_euler("xyz", p1_batch[:, 3:6]).as_matrix()
    R2 = R.from_euler("xyz", p2_batch[:, 3:6]).as_matrix()
    id = np.array([[[0, 0, 0, 1]]]).repeat(n, axis=0)
    T1 = np.concatenate([np.concatenate((R1, t1), axis=2), id], axis=1)
    T2 = np.concatenate([np.concatenate((R2, t2), axis=2), id], axis=1)

    # Process rotations
    for i in range(n):

        T1_inv = np.linalg.inv(T1[i])
        transformations[i, :3] = (T1_inv @ T2[i] @ np.array([0, 0, 0, 1]))[:3]
        t_inv = T2[i] @ T1_inv
        transformations[i, 3:6] = R.from_matrix(t_inv[:-1, :-1]).as_euler("xyz")

    return transformations


if __name__ == "__main__":
    pass
