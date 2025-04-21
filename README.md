# tracking-football-players

A repository for tracking football players

- Player, Referee & Ball Detection + Tracking
- Team possession
- 2D Top Down View
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

*Optional*

If you have a CUDA capable machine, you can install torch with CUDA for faster inference. 
More information on installing torch can be found on the [pytorch page](https://pytorch.org/get-started/locally/)

```console
pip uninstall torch torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

- Run the following command
```console
python main.py
```

- Select your video from the UI
- To run analysis and generate heatmaps, click *Run Analysis*
- To classify a video into an action, click *Run Action Recognition*

- Your output will be saved in *output_videos*