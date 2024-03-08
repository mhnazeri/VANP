from pathlib import Path
import pickle
from typing import Union
import copy

import cv2
from PIL import Image
import numpy as np
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset


def cartesian_to_polar(x: float, y: float) -> torch.Tensor:
    """Convert cartesian coordinates to polar coordinates
    x: (float) x direction
    y: (float) y direction
    """
    # Calculating radius
    radius = math.sqrt(x * x + y * y)
    # Calculating angle (theta) in radian
    theta = math.atan2(y, x)
    return torch.Tensor([radius, theta]).float()


def imread(address: str):
    img = cv2.imread(address, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


class SocialNavDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        resize: Union[list, tuple] = (224, 224),
        metric_waypoint_spacing: float = 1.0,
        only_non_linear: bool = False,
    ):
        """Dataloader for social navigation task

        root (str): root directory of files
        """
        self.resize = resize
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.train = train
        # read and store directories
        with Path(root).open("rb") as f:
            self.data = pickle.load(f)

        # define transformations
        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.resize, antialias=True),
                    transforms.RandomAutocontrast(p=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean hardcoded
                        std=[0.229, 0.224, 0.225],
                    ),  # ImageNet std hardcoded
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.resize, antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        if only_non_linear:
            # select non_linear trajectories
            non_linear_trajs = np.nonzero(self.data["non_linear"])
            self.data["past_positions"] = np.array(self.data["past_positions"])[
                non_linear_trajs
            ]
            self.data["future_positions"] = np.array(self.data["future_positions"])[
                non_linear_trajs
            ]
            self.data["past_yaw"] = np.array(self.data["past_yaw"])[non_linear_trajs]
            self.data["future_yaw"] = np.array(self.data["future_yaw"])[
                non_linear_trajs
            ]
            self.data["past_vw"] = np.array(self.data["past_vw"])[non_linear_trajs]
            self.data["future_vw"] = np.array(self.data["future_vw"])[non_linear_trajs]
            self.data["past_frames"] = np.array(self.data["past_frames"])[
                non_linear_trajs
            ]
            self.data["future_frames"] = np.array(self.data["future_frames"])[
                non_linear_trajs
            ]

    def __len__(self):
        return len(self.data["past_positions"])

    def __getitem__(self, idx):
        """Return a sample in the form: (uuid, image, action)"""
        # Read image data and add to the list
        past_frames = []
        original_frame = (
            copy.deepcopy(
                np.array(imread(str(self.data["past_frames"][idx][-1]))).astype(
                    np.float32
                )
            )
            / 255.0
        )
        future_frame = imread(
            self.data["future_frames"][idx][12].as_posix()
        )  # which future frames is the best? 12 is 3 seconds, -1 is 5 secs
        for img_address in self.data["past_frames"][idx]:
            # read all images and append them to a list
            img = imread(str(img_address))
            img = self.transform(img)
            # img = img.refine_names(..., 'channels', 'height', 'width')
            past_frames.append(img)

        sample = {
            "past_positions": self.data["past_positions"][idx],  # (Obs_len, 2)
            "future_positions": self.data["future_positions"][idx],  # (Pred_len, 2)
            "past_yaw": self.data["past_yaw"][idx],  # (Obs_len, )
            "future_yaw": self.data["future_yaw"][idx],  # (Pred_len, )
            "past_vw": self.data["past_vw"][idx],  # (Obs_len, 2)
            "future_vw": self.data["future_vw"][idx],  # (Pred_len, 2)
            "past_frames": past_frames,
            "original_frame": original_frame,
            "future_frame": self.transform(future_frame),
        }
        current = copy.deepcopy(sample["past_positions"][-1])  # current position
        rot = np.array(
            [
                [np.cos(sample["past_yaw"][-1]), -np.sin(sample["past_yaw"][-1])],
                [np.sin(sample["past_yaw"][-1]), np.cos(sample["past_yaw"][-1])],
            ],
            dtype=np.float32,
        )
        sample["past_positions"] = (sample["past_positions"] - current).dot(
            rot
        ) * self.metric_waypoint_spacing  # these will be behind the ego
        sample["future_positions"] = (sample["future_positions"] - current).dot(
            rot
        ) * self.metric_waypoint_spacing
        # how many steps to each the goal?
        dt = np.random.randint(
            low=len(sample["future_positions"]) // 2,
            high=len(sample["future_positions"]),
        )
        goal = copy.deepcopy(sample["future_positions"][dt])
        goal = cartesian_to_polar(goal[0], goal[1])
        goal = goal / (torch.norm(goal) + 0.000006)  # normalize the goal
        sample["goal_direction"] = goal
        sample["dt"] = torch.Tensor(
            [
                1 - (1 / dt),  # Normalize the time
            ]
        )
        return sample


if __name__ == "__main__":
    pass
