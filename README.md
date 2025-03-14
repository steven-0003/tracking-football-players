# tracking-football-players

A repository for tracking football players

- Player, Referee & Ball Detection + Tracking
- Statistics: Possession, Speed + Distance Covered
- 2D Homography
- Heatmaps
- Action Recognition

## Prerequisites

- Python 3.11 or higher

- Create a venv (see [here](https://docs.python.org/3/library/venv.html)) and install requirements.txt

For example in Windows:

```console
# Create a venv
py -3.11 -m venv .venv

# Activate venv
./.venv/Scripts/Activate.ps1

# Install requirements
pip install -r requirements.txt
```

## Usage

- Create a folder in the root directory called *input_videos* and place your video here
- Replace the *video_name* in *main.py* with the name of your video

- To get the ouput, run the following command
```console
python main.py
```

- Your output will be saved in *output_videos*