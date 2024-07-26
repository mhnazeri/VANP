[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official PyTorch implementation of ["Learning Where to See for Navigation: A Self-Supervised Vision-Action Pre-Training Approach"](https://arxiv.org/abs/2403.08109).

<p align="center">
  <img src="docs/VANP.svg"  height="150" width="600"/>
</p>

## Remarks
* Update: added more augmentation and gradient accumulation for better performance.
* Update: code cleanup, and general bug fixes.
* If you want to apply it on your dataset, please make sure that your data does not contain static sequences for better results. Please read the limitation section of the paper.
* Removing the action head is possible, but generally not advised at least have it for warmup.
* You can change the hyperparameters in the config file, and the level of augmentations in the dataloader to improve the results.


## Installation
Main libraries:
* [PyTorch](https://www.pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files

First create a virtual env for the project. 
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the latest version of PyTorch from the [official site](https://www.pytorch.org/). Finally, run the following:
```bash
pip install -r requirements.txt
```
To set up Comet.Ml follow the [official documentations](https://www.comet.ml/docs/).

## Dataset
Please follow [this](docs/data_parser.md) guide to download the dataset.

## Training
<p align="center">
  <img src="docs/vanp.gif"/>
</p>

To run pretext training (edit [config](VANP/conf/config_pretext.yaml) first) then run:
```bash
./run.sh train
```

[//]: # (## Sample Outputs)

[//]: # (Unlike ImageNet weights which primarily focus on a single salient object within the environment, regardless of its distance, )

[//]: # (the proposed VANP demonstrates greater accuracy in attending to multiple nearby )

[//]: # (objects that directly influence the robot's trajectory by activating regions corresponding to )

[//]: # (pedestrians, cars, trash cans, doors, and other relevant elements.)

[//]: # ()
[//]: # (![Sample outputs]&#40;docs/samples/sample1.jpg&#41;)

[//]: # ()
[//]: # (However, the model sometimes fails to pay attention to the important regions affecting the trajectory. )

[//]: # (We can see activations in the sky or lots of unnecessary activations:)

[//]: # ()
[//]: # (![Sample outputs]&#40;docs/samples/sample2.jpg&#41;)

## Acknowledgements
Thanks for [GNM](https://github.com/PrieureDeSion/drive-any-robot), [VICreg](https://github.com/facebookresearch/vicreg/tree/main), and [Barlow](https://github.com/facebookresearch/barlowtwins) papers for making their code public.

If you find the code helpful, please cite this work:
```
@article{nazeri2024vanp,
  title={VANP: Learning Where to See for Navigation with Self-Supervised Vision-Action Pre-Training},
  author={Nazeri, Mohammad and Wang, Junzhe and Payandeh, Amirreza and Xiao, Xuesu},
  journal={arXiv preprint arXiv:2403.08109},
  year={2024}
}
```
