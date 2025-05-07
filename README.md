# Tracking Football Players

A repository for tracking football players

Builds on this [existing repository](https://github.com/abdullahtarek/football_analysis)

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

Navigate to the root folder and extract the [model zip](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/steven_moussa_student_manchester_ac_uk/ETl8K3SJW9lMggHoPcMxe5gBKj3CyPyDf-0hrl7bt57VBQ?e=lRMoRv).

The directory should be as follows

```
root
|
|___models
    |   best.pt
    |___action
    |       best.pt
    |___keypoints
            best.pt
    main.py
...
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
- To run analysis and generate heatmaps, click *Run Analysis* <br> 
The following [demo video](https://livemanchesterac-my.sharepoint.com/:v:/g/personal/steven_moussa_student_manchester_ac_uk/EZrn88sAWiNJhCXDqiWXuOUBp4NbyaUu4dqm4ptlKnU1uA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=qNnqoe) is provided to run analysis on. Note that for generating heatmaps, it is advisable to run on longer videos.
- To classify a video into an action, click *Run Action Recognition* <br>
The following [demo video](https://livemanchesterac-my.sharepoint.com/:v:/g/personal/steven_moussa_student_manchester_ac_uk/EV7phZXB17pJl9lTHtnHBSMBP-HqF4FewgUAE4VaFFN7hg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rIainF) is provided for action recognition. Note that the video clip must already be cut to where the action begins and ends. A simple action recognition model is used classifying a clip into 4 actions: short pass, dribble, diving, throw

- Your video output and generated heatmaps will be saved in *output/video_name*