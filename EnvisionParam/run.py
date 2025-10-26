from envisionhgdetector import GestureDetector

# Initialize detector with model selection
detector = GestureDetector(
    model_type="lightgbm",      # "cnn" or "lightgbm"  
    motion_threshold=0.5,       # CNN only: sensitivity to motion
    gesture_threshold=0.6,      # Confidence threshold for gestures
    min_gap_s=0.3,             # Minimum gap between gestures (post-hoc)
    min_length_s=0.5,          # Minimum gesture duration (post-hoc)
    gesture_class_bias=0.0     # CNN only: bias toward gesture vs move
)

# Process multiple videos
results = detector.process_folder(
    input_folder="videos_to_label/input.mp4",
    output_folder="output"
)
