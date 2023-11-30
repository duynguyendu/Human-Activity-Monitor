# Human-Activity-Monitor

## Install
> Recommend to use [Conda](https://docs.conda.io/projects/miniconda/en/latest/)

**Python == 3.10**  
```bash
pip install -r requirements.txt
```

## Docker
### Build image
```bash
docker build -t har .
```

### Run
```bash
# CPU only
docker -it --net=host har

# With GPU
docker -it --net=host --gpus all har

# Show video (Linux only)
xhost + && \
docker run -it --rm --net=host (--gpus all) \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix har && \
xhost -
```

## Run
> All configuration can be access at [configs/run](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/run) folder

CLI options:
```bash
python3 src/run.py --help
```

Example:
```bash
python3 src/run.py video.path=data/video/abc.mp4 video.speed=2 detector.model.conf=0.5 classifier=false features.heatmap=false features.track_box=false
```


## Train
### Setup
1. Dataset must be put in the [data](https://github.com/HT0710/Human-Activity-Recognition/tree/main/data) folder
2. Configure the options in [configs/data](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/data) folder
3. Then using the `video_preparation.py` or `image_preparation.py` file in the [tools](https://github.com/HT0710/Human-Activity-Recognition/blob/main/tools) folder to generate the trainable data
```bash
# If using image
python3 tools/image_preparation.py auto.data_path=path/to/the/data

# If using video
python3 tools/video_preparation.py auto.data_path=path/to/the/data
```

### Run
Configure the training setting in [configs/train.yaml](https://github.com/HT0710/Human-Activity-Recognition/blob/main/configs/train.yaml):  
CLI options:
```bash
python3 src/train.py --help
```
Training:
```bash
python3 src/train.py
```

## Note
- Most of the configurations can found in the **[configs](https://github.com/HT0710/Human-Activity-Recognition/tree/main/configs)** folder. `train.py` contains additional setting need to be changed in the file directly.
- See [https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/) for configuration and CLI help.
