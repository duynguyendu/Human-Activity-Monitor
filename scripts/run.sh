#!/bin/bash


# Note:
#   - !!!Enable a feature by remove that line completely
#   - Configuration can be temporary modified here and run with `./scripts/run.sh`
#   - For more configuration: `python3 src/run.py --help` or `./scripts/run-help.sh`

python3 src/run.py video.path=data/video/7_10.mov \
    video.speed=1 \
    record=false \
    classifier=false \
    features.track_box=false \
    features.heatmap=false \
