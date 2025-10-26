from envisionhgdetector import utils
from envisionhgdetector import GestureDetector
import os

output_folder="output"

# Step 1: Cut videos by detected segments
segments = utils.cut_video_by_segments(output_folder)

# Step 2: Set up analysis folders
gesture_segments_folder = os.path.join(output_folder, "gesture_segments")
retracked_folder = os.path.join(output_folder, "retracked")
analysis_folder = os.path.join(output_folder, "analysis")

# Step 3: Retrack gestures with world landmarks
tracking_results = detector.retrack_gestures(
    input_folder=gesture_segments_folder,
    output_folder=retracked_folder
)

# Step 4: Compute DTW distances and kinematic features
analysis_results = detector.analyze_dtw_kinematics(
    landmarks_folder=tracking_results["landmarks_folder"],
    output_folder=analysis_folder
)

# Step 5: Create interactive dashboard
detector.prepare_gesture_dashboard(
    data_folder=analysis_folder
)
# Then run: python app.py (in output folder)
