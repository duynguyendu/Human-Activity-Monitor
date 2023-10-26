# Human-Activity-Recognition

## Install
`pip install -r requirements.txt`

## Structure
- configs: Contains configuration files
- logs: Stores log files
- src: The main source code directory
  - data: Contains dataset
    - ...
  - models: Contains models
    - ...
  - modules: Contains support modules and utilities
    - ...
  - eval.py
  - train.py
  - ...

## Setup
1. Dataset must be put in the [src/data](https://github.com/HT0710/Human-Activity-Recognition/tree/main/src/data) folder
2. Configure the options in [configs/data.yaml](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/data.yaml) file
3. Then using the `data_preparation.py` file to generate the trainable data
```bash
python src/data_preparation.py
```

## Train
Configure the training setting in [configs/train.yaml](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/train.yaml) or run:
```bash
python src/train.py --help
```


### Run
```bash
python src/train.py
```

## Note
- Most of the configurations can found in the **[configs](https://github.com/HT0710/Human-Activity-Recognition/tree/main/configs)** folder. Other like `train.py` need to be changed in the file directly.
- See [https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/) for configuration and CLI help.
