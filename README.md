[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The official implementation of "VANP: A Self-Supervised Vision-Action Model for Visual Navigation Pre-Training".

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
To download and the dataset please follow [this](docs/data_parser.md) guide.

## Training
To run pretext training (edit [config](VANP/conf/config_pretext.yaml) first):
```bash
./run.sh train
```

## Acknowledgements
Thanks for [GNM](https://github.com/PrieureDeSion/drive-any-robot), [VICreg](https://github.com/facebookresearch/vicreg/tree/main), and [Barlow](https://github.com/facebookresearch/barlowtwins) papers for making their code public.
