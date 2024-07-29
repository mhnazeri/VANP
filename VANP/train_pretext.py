from pathlib import Path
from datetime import datetime
import sys
import argparse
from functools import partial
import copy

try:
    sys.path.append(str(Path(".").resolve()))
except Exception as e:
    raise RuntimeError("Can't append root directory of the project to the path") from e

from rich import print
import comet_ml
from comet_ml.integration.pytorch import log_model, watch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns
from tqdm import tqdm
import torch
from torchvision import models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from icecream import ic, install

from model.pretext_model import PretextModel
from model.autoencoder import Autoencoder, Transformer
from VANP.model.policynet import TransformerPolicy, PoolingPolicy, MLPPolicy
from model.model import EndToEnd
from model.dataloader import SocialNavDataset
from utils.nn import (
    check_grad_norm,
    save_checkpoint,
    load_checkpoint,
    init_weights,
    make_mlp,
    get_activation,
)
from utils.helpers import get_conf, timeit
from utils.loss import EndToEndLoss, barlow_loss, vicreg_loss


class Learner:
    def __init__(self, cfg_dir: str):
        # load config file and initialize the logger and the device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.logger.experiment_name
        self.cfg.directory.model_name += f"-{self.cfg.model.img_backbone.name}-{self.cfg.train_params.loss}-{'pretrained' if self.cfg.model.img_backbone.pretrained else 'random'}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        if self.cfg.train_params.debug:
            install()
            ic.enable()
            ic.configureOutput(prefix=lambda: f"{datetime.now():%Y-%m-%d %H:%M:%S} |> ")
            torch.autograd.set_detect_anomaly(True)
            self.cfg.logger.disabled = True
        else:
            ic.disable()
            torch.autograd.set_detect_anomaly(True)
            matplotlib.use("Agg")
        self.logger = self.init_logger(self.cfg.logger)
        self.device = self.init_device()
        # fix the seed for reproducibility
        torch.random.manual_seed(self.cfg.train_params.seed)
        torch.cuda.manual_seed(self.cfg.train_params.seed)
        torch.cuda.manual_seed_all(self.cfg.train_params.seed)
        torch.backends.cudnn.benchmark = True
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model, self.pretext_model = self.init_model(self.cfg.model)
        self.update_weights()
        self.logger.log_code(folder="./VANP/model")
        watch(self.pretext_model, log_step_interval=100)
        num_params = [x.numel() for x in self.model.parameters()]
        trainable_params = [
            x.numel() for x in self.model.parameters() if x.requires_grad
        ]
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of parameters: {sum(num_params) / 1e6:.2f}M")
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of trainable parameters: {sum(trainable_params) / 1e6:.2f}M"
        )
        # initialize the optimizer
        ## remove duplicate terms
        params = set(self.model.parameters()) | set(self.pretext_model.parameters())
        self.optimizer = torch.optim.AdamW(
            # list(self.model.parameters()) + list(self.pretext_model.parameters()),
            filter(lambda p: p.requires_grad, self.pretext_model.parameters()),
            **self.cfg.adamw)
        self.optimizer_dt = torch.optim.SGD(
            # list(self.model.parameters()) + list(self.pretext_model.parameters()),
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.sgd)
        # criterion
        self.criterion = (
            partial(barlow_loss, **self.cfg.barlow_loss)
            if self.cfg.train_params.loss == "barlow"
            else partial(vicreg_loss, **self.cfg.vicreg_loss)
        )
        self.criterion_dt = EndToEndLoss(
            self.cfg.train_params.loss_type_dt, self.cfg.train_params.temp_loss_lambda
        )
        # initialize the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dt, T_max=self.cfg.train_params.epochs
        )
        # if resuming, load the checkpoint
        self.if_resume()

    def train(self):
        """Trains the model"""

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []

            # unfreeze the weights
            if self.cfg.train_params.unfreeze_after == self.epoch:
                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} - UNFREEZING action encoder!"
                )
                for (
                    name,
                    child,
                ) in self.pretext_model.action_backbone.named_parameters():
                    child.requires_grad_(True)
                self.pretext_model.action_backbone.train()
            self.pretext_model.requires_grad_(
                True
            )  # making sure the weights are updating
            self.pretext_model.train()
            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
            )
            for data in bar:
                self.iteration += 1
                with self.logger.train():
                    (loss, repr_loss, std_loss, cov_loss,grad_norm), t_train = (
                        self.forward_batch(data)
                    )
                    t_train /= self.data.batch_size
                    running_loss.append(loss)

                    bar.set_postfix(
                        loss=loss,
                        Grad_Norm=grad_norm,
                        Time=t_train,
                        repr_loss=repr_loss,
                        std_loss=std_loss,
                        cov_loss=cov_loss
                    )

                    self.logger.log_metrics(
                        {
                            "batch_loss": loss,
                            "grad_norm": grad_norm,
                            "repr_loss": repr_loss,
                            "std_loss": std_loss,
                            "cov_loss": cov_loss
                        },
                        epoch=self.epoch,
                        step=self.iteration,
                    )
            # update learning rate
            # self.scheduler.step()
            if self.epoch % self.cfg.train_params.visualize_every == 0:
                with torch.no_grad():
                    self.pretext_model.eval()
                    activation = {}
                    self.pretext_model.img_backbone.layer4[-1].relu.register_forward_hook(
                        get_activation(activation, "conv3")
                    )
                    past_frames = [x.to(self.device) for x in data["past_frames"]]
                    _ = self.pretext_model.img_backbone(past_frames[-1])
                    for idx in range(data["future_positions"].size(0) // 10):
                        attentions = activation["conv3"][idx].unsqueeze(0).detach()
                        attentions = np.mean(
                            torch.nn.functional.interpolate(
                                attentions, scale_factor=(data['original_frame'][idx].shape[-3]/attentions.shape[-2], data['original_frame'][idx].shape[-2]/attentions.shape[-1]), mode="bicubic"
                            )
                            .squeeze()
                            .cpu()
                            .numpy(),
                            axis=0,
                        )

                        plt.imshow(data["original_frame"][idx].cpu())
                        plt.imshow(attentions, alpha=0.6, cmap="hot")
                        plt.axis("off")
                        self.logger.log_figure(
                            f"sample_E{self.epoch:03}|S{idx:03}_image",
                            plt,
                            step=self.iteration,
                        )
                        plt.clf()
                        plt.close()
                        full_images = list(img[idx] for img in data["past_frames"])
                        full_images.append(data["future_frame"][idx])
                        full_images = make_grid(full_images)
                        self.logger.log_image(
                            full_images,
                            name=f"sample_E{self.epoch:03}|S{idx:03}_fullimage",
                            step=self.iteration,
                            image_channels="first",
                        )

            bar.close()
            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: {self.e_loss[-1]:.2f}"
            )

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            # train downstream task every train_dt epochs
            if self.epoch % self.cfg.train_params.train_dt == 0:
                # update the model
                self.update_weights()
                self.model.train()
                for i in range(10):  # train downstream head for 10 epochs
                    # load data again
                    bar = tqdm(
                        self.data,
                        desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training downstream: ",
                    )
                    running_loss_dt = []
                    for data in bar:
                        self.iteration += 1
                        (loss, grad_norm), t_train = self.forward_batch_dt(data)
                        t_train /= self.data.batch_size
                        running_loss_dt.append(loss)

                        bar.set_postfix(
                            LossDT=loss, Grad_NormDT=grad_norm, Time=t_train
                        )

                        self.logger.log_metrics(
                            {
                                "batch_lossDT": loss,
                                "grad_normDT": grad_norm,
                            },
                            epoch=self.epoch,
                            step=self.iteration,
                        )

                    bar.close()
                # validate on val set
                (val_loss3s, val_loss5s), t = self.validate()
                t /= len(self.val_data.dataset)

                # average downstream loss for an epoch
                self.e_loss_dt.append(np.mean(running_loss_dt))  # epoch loss
                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                    + f"Iteration {self.iteration:05} summary: "
                    + f"Val loss3s: {val_loss3s:.2f}"
                    + f"\t| Val loss5s: {val_loss5s:.2f}"
                    + f"\t| time: {t:.3f} seconds\n"
                )

                self.logger.log_metrics(
                    {
                        "train_lossDT": self.e_loss_dt[-1],
                        "val_loss3s": val_loss3s,
                        "val_loss5s": val_loss5s,
                        "time": t,
                    },
                    epoch=self.epoch,
                    step=self.iteration,
                )

                if self.epoch % self.cfg.train_params.save_every == 0 or (
                    self.e_loss_dt[-1] <= self.best
                    and self.epoch >= self.cfg.train_params.start_saving_best
                ):
                    self.save()

            self.epoch += 1
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.pretext_model.train()
        # move data to device

        past_frames = [x.to(self.device) for x in data["past_frames"]]
        future_frame = data["future_frame"].to(device=self.device)
        # select action type
        if self.cfg.model.action_type == "vw":
            future_actions = data["future_vw"].to(device=self.device)
            future_actions[:, :, 1] = (
                future_actions[:, :, 1] * self.cfg.train_params.vw_scaling_factor
            )
        elif self.cfg.model.action_type == "xy":
            future_actions = data["future_positions"].to(device=self.device)
            if self.cfg.model.action_size == 3:
                future_actions = torch.cat(
                    [
                        future_actions,
                        data["future_yaw"].to(device=self.device).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            elif self.cfg.model.action_size == 4:
                future_actions = torch.cat(
                    [
                        future_actions,
                        torch.sin(data["future_yaw"])
                        .to(device=self.device)
                        .unsqueeze(-1),
                        torch.cos(data["future_yaw"])
                        .to(device=self.device)
                        .unsqueeze(-1),
                    ],
                    dim=-1,
                )

        if self.cfg.model.action_encoder_type in "mlp":
            future_actions = future_actions.view(future_actions.shape[0], -1)

        # forward, backward
        _, img_z, action_z, future_frame_z = self.pretext_model(
            past_frames, future_frame, future_actions
        )
        loss_vision, (repr_loss, std_loss, cov_loss) = self.criterion(img_z, future_frame_z)
        loss_action, (repr_loss_a, std_loss_a, cov_loss_a) = self.criterion(img_z, action_z)
        loss = (
            self.cfg.train_params.loss_lambda * loss_vision
            + (1 - self.cfg.train_params.loss_lambda) * loss_action
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.pretext_model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        if (self.iteration + 1) % self.cfg.train_params.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # check grad norm for debugging
        grad_norm = check_grad_norm(self.pretext_model)
        return loss.detach().item(), repr_loss.item() + repr_loss_a.item(), std_loss.item() + std_loss_a.item(), cov_loss.item() + cov_loss_a.item(), grad_norm

    @timeit
    def forward_batch_dt(self, data):
        """Forward pass of a batch"""
        self.model.train()
        ## select only the last obs_len frames
        ic(len(data["past_frames"]))
        past_frames = [x.to(self.device) for x in data["past_frames"]][
            -self.cfg.model.obs_len :
        ]
        goal_direction = data["goal_direction"].to(device=self.device)
        dt_gt = data["dt"].to(device=self.device)
        if self.cfg.model.action_type == "vw":
            future_actions = data["future_vw"].to(device=self.device)
            future_actions[:, :, 1] = (
                future_actions[:, :, 1] * self.cfg.train_params.vw_scaling_factor
            )
        elif self.cfg.model.action_type == "xy":
            future_actions = data["future_positions"].to(device=self.device)
            if self.cfg.model.action_size == 3:
                future_actions = torch.cat(
                    [
                        future_actions,
                        data["future_yaw"].to(device=self.device).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            elif self.cfg.model.action_size == 4:
                future_actions = torch.cat(
                    [
                        future_actions,
                        torch.sin(data["future_yaw"])
                        .to(device=self.device)
                        .unsqueeze(-1),
                        torch.cos(data["future_yaw"])
                        .to(device=self.device)
                        .unsqueeze(-1),
                    ],
                    dim=-1,
                )
        ic(future_actions.size())
        ic(future_actions[:, :, 0].mean())
        ic(torch.abs(future_actions[:, :, 1]).mean())

        action, _ = self.model(past_frames, goal_direction)
        _, dt = self.model(past_frames, goal_direction)

        loss = self.criterion_dt(future_actions, action, dt_gt, dt)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient clipping
        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        # update
        self.optimizer_dt.step()
        # check grad norm for debugging
        grad_norm = check_grad_norm(self.model)

        return loss.detach().item(), grad_norm

    @timeit
    @torch.no_grad()
    def validate(self):
        self.model.eval()

        running_loss3s = []
        running_loss5s = []
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating",
        )
        activation = {}
        hook_handler = self.model.image_encoder.layer4[-1].relu.register_forward_hook(get_activation(activation, 'conv3'))
        for data in bar:
            # move data to device
            past_frames = [x.to(self.device) for x in data["past_frames"]][
                -self.cfg.model.obs_len :
            ]
            goal_direction = data["goal_direction"].to(self.device)
            if self.cfg.model.action_type == "vw":
                future_actions = data["future_vw"].to(device=self.device)
                future_actions[:, :, 1] = (
                    future_actions[:, :, 1] * self.cfg.train_params.vw_scaling_factor
                )

            elif self.cfg.model.action_type == "xy":
                future_actions = data["future_positions"].to(device=self.device)
                if self.cfg.model.action_size == 3:
                    future_actions = torch.cat(
                        [
                            future_actions,
                            data["future_yaw"].to(device=self.device).unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                elif self.cfg.model.action_size == 4:
                    future_actions = torch.cat(
                        [
                            future_actions,
                            torch.sin(data["future_yaw"])
                            .to(device=self.device)
                            .unsqueeze(-1),
                            torch.cos(data["future_yaw"])
                            .to(device=self.device)
                            .unsqueeze(-1),
                        ],
                        dim=-1,
                    )
            # future_actions = data["future_vw"].to(self.device)
            ic(future_actions.size())
            # forward
            pred_actions, _ = self.model(past_frames, goal_direction)
            # check loss in 5 seconds
            loss5s = torch.nn.functional.mse_loss(pred_actions, future_actions)
            # only 3 seconds
            loss3s = torch.nn.functional.mse_loss(
                pred_actions[:, :12, :], future_actions[:, :12, :]
            )
            running_loss5s.append(loss5s.item())
            running_loss3s.append(loss3s.item())
            bar.set_postfix(loss3s=loss3s.item(), loss5s=loss5s.item())
        hook_handler.remove()
        bar.close()

        # average loss
        loss3s = np.mean(running_loss3s)
        loss5s = np.mean(running_loss5s)
        if self.cfg.model.action_type == "vw":
            pred_actions[:, :, 1] /= self.cfg.train_params.vw_scaling_factor
            future_actions[:, :, 1] /= self.cfg.train_params.vw_scaling_factor
        elif self.cfg.model.action_type == "xy":
            if self.cfg.model.action_size == 4:
                future_actions[:, :, 2:] = torch.atan2(
                    torch.sin(data["future_yaw"]).to(device=self.device).unsqueeze_(-1),
                    torch.cos(data["future_yaw"]).to(device=self.device).unsqueeze_(-1),
                )
                pred_actions[:, :, 2:] = torch.atan2(
                    torch.sin(pred_actions[:, :, 2])
                    .to(device=self.device)
                    .unsqueeze_(-1),
                    torch.cos(pred_actions[:, :, 3])
                    .to(device=self.device)
                    .unsqueeze_(-1),
                )

            # visualize the output
            for idx in range(past_frames[-1].size(0) // 4):
                # visualize activations
                attentions = activation["conv3"][idx].unsqueeze(0).detach()
                attentions = np.mean(
                    torch.nn.functional.interpolate(
                        attentions, scale_factor=(data['original_frame'][idx].shape[-3]/attentions.shape[-2], data['original_frame'][idx].shape[-2]/attentions.shape[-1]), mode="bicubic"
                    )
                    .squeeze()
                    .cpu()
                    .numpy(),
                    axis=0,
                )

                plt.imshow(data["original_frame"][idx])
                plt.imshow(attentions, alpha=0.6, cmap="hot")
                plt.axis("off")
                self.logger.log_figure(
                    figure_name=f"sample_E{self.epoch:03}|S{idx:03}_img_val",
                    figure=plt,
                    step=self.iteration,
                )
                plt.clf()
            plt.close()

        ic(pred_actions.size())
        ic(future_actions.size())

        return loss3s, loss5s

    def init_device(self):
        """Initializes the device"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the device!")
        is_cuda_available = torch.cuda.is_available()
        device = self.cfg.train_params.device

        if "cpu" in device:
            print(f"Performing all the operations on CPU.")
            return torch.device(device)

        elif "cuda" in device:
            if is_cuda_available:
                device_idx = device.split(":")[1]
                if device_idx == "a":
                    print(
                        f"Performing all the operations on CUDA; {torch.cuda.device_count()} devices."
                    )
                    self.cfg.dataloader.batch_size *= torch.cuda.device_count()
                    return torch.device(device.split(":")[0])
                else:
                    print(f"Performing all the operations on CUDA device {device_idx}.")
                    return torch.device(device)
            else:
                print("CUDA device is not available, falling back to CPU!")
                return torch.device("cpu")
        else:
            raise ValueError(f"Unknown {device}!")

    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        dataset = SocialNavDataset(**self.cfg.dataset)
        val_dataset = SocialNavDataset(**self.cfg.val_dataset)
        # creating dataset interface and dataloader for val data
        if self.cfg.train_params.debug:
            dataset = Subset(dataset, list(range(self.cfg.dataloader.batch_size * 2)))
            val_dataset = Subset(
                val_dataset, list(range(self.cfg.dataloader.batch_size * 2))
            )

        data = DataLoader(dataset, **self.cfg.dataloader)
        self.cfg.dataloader.update({"shuffle": False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        # log dataset status
        self.logger.log_parameters(
            {"train_len": len(dataset), "val_len": len(val_dataset)}
        )
        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples."
        )

        return data, val_data

    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.e_loss_dt = checkpoint["e_loss_dt"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
            self.e_loss = []
            self.e_loss_dt = []

        self.logger.set_epoch(self.epoch)

    def init_logger(self, cfg):
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the logger!")
        logger = None
        # Check to see if there is a key in environment:
        EXPERIMENT_KEY = cfg.experiment_key

        # First, let's see if we continue or start fresh:
        CONTINUE_RUN = cfg.resume
        if EXPERIMENT_KEY and CONTINUE_RUN:
            # There is one, but the experiment might not exist yet:
            api = comet_ml.API()  # Assumes API key is set in config/env
            try:
                api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
            except Exception:
                api_experiment = None
            if api_experiment is not None:
                CONTINUE_RUN = True
                # We can get the last details logged here, if logged:
                # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
                # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

        if CONTINUE_RUN:
            # 1. Recreate the state of ML system before creating experiment
            # otherwise it could try to log params, graph, etc. again
            # ...
            # 2. Setup the existing experiment to carry on:
            logger = comet_ml.ExistingExperiment(
                previous_experiment=EXPERIMENT_KEY,
                log_env_details=self.cfg.logger.log_env_details,  # to continue env logging
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            # Retrieved from above APIExperiment
            # self.logger.set_epoch(epoch)

        else:
            # 1. Create the experiment first
            #    This will use the COMET_EXPERIMENT_KEY if defined in env.
            #    Otherwise, you could manually set it here. If you don't
            #    set COMET_EXPERIMENT_KEY, the experiment will get a
            #    random key!
            if cfg.online:
                logger = comet_ml.Experiment(
                    disabled=cfg.disabled,
                    project_name=cfg.project,
                    log_env_details=self.cfg.logger.log_env_details,
                    log_env_gpu=True,  # to continue GPU logging
                    log_env_cpu=True,  # to continue CPU logging
                    auto_histogram_weight_logging=True,
                    auto_histogram_gradient_logging=True,
                    auto_histogram_activation_logging=True,
                )
                logger.set_name(cfg.experiment_name)
                logger.add_tags(cfg.tags.split())
                logger.log_parameters(self.cfg)
            else:
                logger = comet_ml.OfflineExperiment(
                    disabled=cfg.disabled,
                    project_name=cfg.project,
                    offline_directory=cfg.offline_directory,
                    auto_histogram_weight_logging=True,
                    log_env_details=self.cfg.logger.log_env_details,
                    log_env_gpu=True,  # to continue GPU logging
                    log_env_cpu=True,  # to continue CPU logging
                )
                logger.set_name(cfg.experiment_name)
                logger.add_tags(cfg.tags.split())
                logger.log_parameters(self.cfg)

        return logger

    def save(self, name=None):
        model = self.pretext_model
        if isinstance(self.pretext_model, torch.nn.DataParallel):
            model = model.module

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "visual_encoder": model.img_backbone.state_dict(),
            "action_encoder": model.action_backbone.state_dict(),
            "transformer_encoder": model.transformer_encoder.state_dict(),
            "img_compressor": model.image_compressor.state_dict(),
            "action_compressor": model.action_compressor.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "best": self.best,
            "e_loss": self.e_loss,
            "e_loss_dt": self.e_loss_dt,
            "e2e_model": self.model.state_dict(),
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}-E{self.epoch}"
        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)

    def init_model(self, cfg):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")
        # action data encoder
        if cfg.action_encoder_type == "mlp":
            cfg.action_backbone.dims.insert(0, cfg.pred_len * cfg.action_size)
            action_encoder = make_mlp(**cfg.action_backbone)
            # reverse the dimensions for the decoder
            cfg.action_backbone.dims = list(reversed(cfg.action_backbone.dims))
            action_decoder = make_mlp(**cfg.action_backbone)
            # defining the model
            model = Autoencoder(action_encoder, action_decoder)
            if cfg.action_backbone_weights:
                weights = torch.load(
                    cfg.action_backbone_weights, map_location=self.device
                )["model"]
                model.load_state_dict(weights)
            action_encoder = model.encoder

        elif cfg.action_encoder_type == "attn":
            action_encoder = Transformer(
                n_layers=cfg.attn.num_layers,
                d_model=cfg.attn.context_size,
                n_head=cfg.attn.nhead,
                n_action=cfg.action_size,
                d_hidden=cfg.attn.d_hid,
                pred_len=cfg.pred_len,
                action_type=cfg.action_type,
                dropout=cfg.attn.dropout,
                n_registers=cfg.attn.n_registers,
            ).to(self.device)

            if cfg.action_backbone_weights:
                weights = torch.load(
                    cfg.action_backbone_weights, map_location=self.device
                )["model"]
                action_encoder.load_state_dict(weights)

        # freeze the weights
        if cfg.freeze_action_backbone:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - FREEZING action encoder!")
            for name, child in action_encoder.named_parameters():
                child.requires_grad_(False)
            action_encoder.eval()

        if "resnet" in cfg.img_backbone.name.lower():
            weights = "DEFAULT" if self.cfg.model.img_backbone.pretrained else None
            print(
                f"Using {cfg.img_backbone.name.upper()} as image backbone with weights {weights}!"
            )
            image_encoder = models.get_model(
                cfg.img_backbone.name.lower(), weights=weights, zero_init_residual=True
            ).to(self.device)
            image_encoder.fc = torch.nn.Identity()

        if not self.cfg.model.img_backbone.pretrained:
            image_encoder.apply(init_weights(**self.cfg.init_model))

        image_encoder.device = self.device
        pretext_model = PretextModel(
            image_encoder,
            action_encoder,
            action_encoder_type=cfg.action_encoder_type,
            feature_size=cfg.feature_size,
            projection_dim=cfg.projection_dim,
            hidden_dim=cfg.hidden_dim,
            lamda=cfg.lamda,
            corr_neg_one=cfg.corr_neg_one,
        )

        # local goal encoder
        self.cfg.model_downstream.goal_encoder.dims.insert(0, 2)  # (x, y) as goal
        self.cfg.model_downstream.goal_encoder.dims.append(
            self.cfg.model_downstream.obs_context_size
        )
        goal_encoder = make_mlp(**self.cfg.model_downstream.goal_encoder).to(
            self.device
        )
        goal_encoder.apply(init_weights(**self.cfg.init_model))
        self.cfg.model_downstream.controller.dims.insert(
            0, self.cfg.model_downstream.obs_context_size
        )
        self.cfg.model_downstream.controller.dims.append(
            self.cfg.model_downstream.action_size * self.cfg.model_downstream.pred_len
        )
        controller = make_mlp(**self.cfg.model_downstream.controller).to(self.device)
        controller.apply(init_weights(**self.cfg.init_model))
        print(
            f"Using {self.cfg.model_downstream.image_encoder.name.upper()} as image backbone!"
        )

        if self.cfg.model_downstream.policy.lower() == "attn":
            # initialize the transformer policy
            controller = TransformerPolicy(
                controller,
                d_model=self.cfg.model_downstream.obs_context_size,
                nhead=self.cfg.model_downstream.nhead,
                d_hid=self.cfg.model_downstream.d_hid,
                num_layers=self.cfg.model_downstream.num_layers,
                dropout=self.cfg.model_downstream.dropout,
                obs_len=self.cfg.model_downstream.obs_len,
                n_registers=self.cfg.model_downstream.n_registers,
            )
        elif self.cfg.model_downstream.policy == "avg":
            controller = PoolingPolicy(
                controller, self.cfg.model_downstream.obs_context_size
            )

        elif self.cfg.model_downstream.policy == "mlp":
            controller = MLPPolicy(
                controller,
                self.cfg.model_downstream.obs_context_size,
                self.cfg.model_downstream.d_hid,
                self.cfg.model_downstream.num_layers,
                self.cfg.model_downstream.dropout,
                self.cfg.model_downstream.obs_len,
            )

        # removing these config keys since they're no longer needed
        image_encoder.device = self.device
        self.cfg.finetune = self.cfg.model_downstream.image_encoder.freeze_weights
        del self.cfg.model_downstream.image_encoder
        del self.cfg.model_downstream.goal_encoder
        del self.cfg.model_downstream.controller
        model = EndToEnd(
            image_encoder, goal_encoder, controller, **self.cfg.model_downstream
        )
        model.image_compressor = copy.deepcopy(pretext_model.image_compressor)

        if (
            "cuda" in str(self.device)
            and self.cfg.train_params.device.split(":")[1] == "a"
        ):
            pretext_model = torch.nn.DataParallel(model)

        return model.to(self.device), pretext_model.to(self.device)

    @torch.no_grad()
    def update_weights(self):
        self.model.image_encoder.load_state_dict(self.pretext_model.img_backbone.state_dict(), strict=False)
        if self.cfg.finetune:
            self.model.image_encoder.requires_grad_(False)

        self.model.image_compressor.load_state_dict(self.pretext_model.image_compressor.state_dict(), strict=False)
        if self.cfg.finetune:
            self.model.image_compressor.requires_grad_(False)

        self.model.controller.ctx_token_emb.copy_(self.pretext_model.ctx_token_emb)
        if self.cfg.finetune:
            self.model.controller.ctx_token_emb.requires_grad_(False)
        # copy the weights of the transformer as well, but don't freeze the weights
        self.model.controller.transformer_encoder.load_state_dict(self.pretext_model.transformer_encoder.state_dict(),
                                                                  strict=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/config", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()
