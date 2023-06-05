# Pitch Overlay 2023
This is a project created with the intention of overlaying pitches from baseballsavant.mlb.com. Under the statcast search function, you can find short videos (<15s) of pitches from pitchers. It's best to overlay pitches from the same at-bat.

![coleoverlaygif](https://github.com/nakschou/PitchOverlay/assets/87676813/1b0dccf9-c72e-4259-86f2-860e2c81edd2)

## Setup:
1. Git clone the repo ```git clone https://github.com/nakschou/OpenCV2023.git```
2. Navigate into the folder you cloned it as, and create your virtual environment. ```pip install -r requirements.txt```. This may take some time.

## How to Use:
1. Begin by going into your config and setting necessary parameters. They're all explained in there, somewhere (I hope).
2. ```python runner.py``` to run the program.
3. Navigate to the first frame on which the pitcher releases the ball, and save that frame.
4. Do the same for the other pitch.
5. Let the model work its (somewhat accurate) magic.

Note that this project is still in development and could result in bad detections. You can try to play with the parameters in config, but these are ultimately going to be fixed by a more robust model.
