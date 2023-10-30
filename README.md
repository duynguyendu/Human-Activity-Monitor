# Human-Activity-Recognition

## Install
> Recommended to use [Conda](https://docs.conda.io/projects/miniconda/en/latest/)

**Python == 3.10**  
```bash
pip install -r requirements.txt
```

## Structure
.  
├── configs  
├── data  
├── logs  
├── scripts  
├── src  
│   ├── components  
│   ├── models  
│   └── modules  
├── tools  
└── weights  

## Setup
1. Dataset must be put in the [data](https://github.com/HT0710/Human-Activity-Recognition/tree/main/data) folder
2. Configure the options in [configs/data.yaml](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/data.yaml) file
3. Then using the `data_preparation.py` file to generate the trainable data
```bash
python src/data_preparation.py
```

## Train
Configure the training setting in [configs/train.yaml](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/train.yaml):  
CLI options:
```bash
python src/train.py --help
```
Training:
```bash
python src/train.py
```

## Note
- Most of the configurations can found in the **[configs](https://github.com/HT0710/Human-Activity-Recognition/tree/main/configs)** folder. `train.py` contains additional setting need to be changed in the file directly.
- See [https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/) for configuration and CLI help.
