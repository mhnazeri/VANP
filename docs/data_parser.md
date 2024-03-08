## Downloading Datasets
To download SCAND please follow [this](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH) link and download the bag files. And download MuSoHu bag files from [here](https://dataverse.orc.gmu.edu/dataset.xhtml?persistentId=doi:10.13021/orc2020/HZI4LJ).

## Installing Dependencies
The parser requires `Python>=3.9` for type annotations. `pyntcloud` package works best with `pandas==2.0.1`.

## Parsing the data

It is recommended to place all the bag files inside the [data](../social_nav/data) directory as depicted in the project structure below. Otherwise, you need to change `bags_dir` variable in the [parser config file](../social_nav/conf/parser.yaml). You can also change other parameters inside the config files.

Project structure:
```
 .
├──  docs
│   └──  data_parser.md
├──  LICENSE.txt
├──  README.md
├──  requirements.txt
├──  run.sh
└──  social_nav
    ├──  __init__.py
    ├──  conf
    │   ├──  config.yaml
    │   ├──  musohu_parser.yaml
    │   └──  scand_parser.yaml
    ├──  data
    │   ├──  musohu
    │   │   └──  03112023_mn_dc_night_1_casual.bag
    │   ├──  processed
    │   │   ├──  A_Jackal_AHG_Library_Thu_Nov_4_16_0
    │   │   │   ├──  point_cloud
    │   │   │   ├──  rgb
    │   │   │   └──  traj_data.pkl
    │   │   ├──  mn_dc_night_1_casual_0
    │   │   │   ├──  depth
    │   │   │   ├──  point_cloud
    │   │   │   ├──  rgb
    │   │   │   └──  traj_data.pkl
    │   └──  scand
    │       └──  A_Jackal_AHG_Library_Thu_Nov_4_16.bag
    ├──  main.py
    ├──  misc_files
    │   └──  FUTURAM.ttf
    ├──  model
    │   ├──  __init__.py
    │   ├──  data_loader.py
    │   └──  net.py
    └──  utils
        ├──  __init__.py
        ├──  helpers.py
        ├──  io.py
        ├──  musohu_parser.py
        ├──  nn.py
        ├──  parser.py
        ├──  parser_utils.py
        ├──  scand_parser.py
        └──  vision.py

```

To run the MuSoHu parser, from the root directory of the project run:
```bash
./run.sh parser musohu
```

And to run SCAND parser, change the parser argument to `scand` like:
```bash
./run.sh parser scand
```
We only store the front facing camera for the Spot in SCAND, so both MuSoHu and SCAND have the *same* interface. The only difference is that SCAND does not contain depth data.

## Creating Samples
To create samples from parsed bags, run the following command:
```bash
./run.sh parser sampler
```
Sampler uses the [parser config](../social_nav/conf/parser.yaml) file to create samples. The sampler uses the observation length (`obs_len`) and prediction length (`pred_len`) from the parser config file to create samples. It uses the directory for storing the pickle file from `parsed_dir`.