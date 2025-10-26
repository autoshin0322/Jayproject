import os
import glob as glob
from pathlib import Path
import pandas as pd
from envisionhgdetector import utils
from envisionhgdetector import GestureDetector
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

currentdir = Path.cwd()
print(f"Current directory: {currentdir}")

# use path join
videofoldertoday = os.path.join(currentdir, 'videos_to_label')
outputfolder = os.path.join(currentdir, 'output')
print(f"Video folder: {videofoldertoday}")
print(f"Output folder: {outputfolder}")

# Initialize detector with model selection
detector = GestureDetector(motion_threshold=0.8, gesture_threshold=0.5, min_gap_s =0.2, min_length_s=0.3, model_type="lightgbm")

# Process multiple videos
results = detector.process_folder(
    input_folder="videos_to_label/input.mp4",
    output_folder="output"
)


detector.process_folder(input_folder=videofoldertoday, output_folder=outputfolder,)
