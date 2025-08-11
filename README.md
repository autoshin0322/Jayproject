# 📖 Noname
👨‍🎓 Goethe-Universität Frankfurt am Main \
🏛️ Fachbereich 12 Institut für Informatik \
📫 E-mail: S6010479@stud.uni-frankfurt.de

## Title: Noname

Betreuer: Prof. Dr. A*** M***, Dr. A*** L***, Dr. A*** H***

🕵️ Envisionhgdetector: https://envisionbox.org/embedded_UsingEnvisionHGdetector_package.html \
📚 Pythonlibrary: https://pypi.org/project/envisionhgdetector \
🔗 Github: https://github.com/WimPouw/envisionhgdetector

💿 Dataset:
1. **TTLab Goethe-Universität - multiperspektive Videos wenden an Dr. H***** https://www.texttechnologylab.org/team/alexander-henlein/
2. **University of Edinburgh, Centre for Language Evolution** https://datashare.ed.ac.uk/handle/10283/3191
3. **IFADV** https://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFADVcorpus/


### Schritt:
1. Mit einem Tool für Binäre Klassifikation erkennen Gesten bzw. BIO-Label oder Gesten, Non-Gesten
2. Datensatz: R-Daten aus Va.Si.Li-Lab verwendet werden. Dazu gehören die Multiperspektiven-Videos 
3. Tool auswählen : Envisionhgdetector (Empfehlung von Dr. Henlein)
4. Ausführen den Envisionhgdetector mit Datensatz, ob er Gesten gut erkennt.
5. noch offen

***

## Aktueller Stand
1. envisionhgdetector testen
  - [x] TTLab Dataset
  - [x] University of Edinburgh Dataset

### Lets get started
```python
import os
import glob as glob
from pathlib import Path

videoname="D_21"

currentdir = Path.cwd()
print(f"Current directory: {currentdir}")

videofoldertoday = os.path.join(currentdir, f'{videoname}/videos_to_label')
outputfolder = os.path.join(currentdir, f'{videoname}/output')
print(f"Video folder: {videofoldertoday}")
print(f"Output folder: {outputfolder}")
```
Current directory: /Users/jaehyunshin/Desktop/envision
Video folder: /Users/jaehyunshin/Desktop/envision/D_21/videos_to_label
Output folder: /Users/jaehyunshin/Desktop/envision/D_21/output
```python
import glob
from IPython.display import Video

# List all videos in the folder
videos = glob.glob(os.path.join(videofoldertoday, '*.mp4'))
# Display single video
Video(videos[0], embed=True, width=300)
```
video
```python
from envisionhgdetector import GestureDetector
import os

videoname="D_21"

# absolute path 
videofoldertoday = os.path.abspath(f'{videoname}/videos_to_label/')
outputfolder = os.path.abspath(f'{videoname}/output/')

# create a detector object (note that the motion threshold is not active in LightGBM, so we just set it to same value as gesture threshold)
detector = GestureDetector(motion_threshold=0.5, gesture_threshold=0.6, min_gap_s =0.3, min_length_s=0.5, model_type="lightgbm")

# just do the detection on the folder
detector.process_folder(
    input_folder=videofoldertoday,
    output_folder=outputfolder,
)
```
Loading LightGBM model from /opt/homebrew/Caskroom/miniconda/base/envs/envision/lib/python3.10/site-packages/envisionhgdetector/model/lightgbm_gesture_model_v1.pkl
LightGBM model loaded successfully!
Window size: 5 frames
Available gestures: ['Move', 'NoGesture', 'Gesture']
Advanced features: ENABLED
Expected features: 80
Initialized LightGBM gesture detector

Processing hair_salon.mp4 with LIGHTGBM model...
Extracting features and model inferencing...
Processing video with LightGBM: 30.0fps, 155 frames
I0000 00:00:1754921343.468372 16297451 gl_context.cc:369] GL version: 2.1 (2.1 Metal - 88), renderer: Apple M1 Pro
W0000 00:00:1754921343.578306 16354099 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1754921343.589766 16354100 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Generating labeled video...

Labeling video:   0%|                                                                                                       | 0/129 [00:00<?, ?frames/s]
Labeling video:   9%|███████▉                                                                                     | 11/129 [00:00<00:01, 103.92frames/s]
Labeling video:  17%|████████████████                                                                              | 22/129 [00:00<00:01, 98.87frames/s]
Labeling video:  25%|███████████████████████▎                                                                      | 32/129 [00:00<00:01, 91.43frames/s]
Labeling video:  33%|██████████████████████████████▌                                                               | 42/129 [00:00<00:01, 83.11frames/s]
Labeling video:  40%|█████████████████████████████████████▏                                                        | 51/129 [00:00<00:01, 76.04frames/s]
Labeling video:  46%|██████████████████████████████████████████▉                                                   | 59/129 [00:00<00:01, 69.15frames/s]
Labeling video:  52%|████████████████████████████████████████████████▊                                             | 67/129 [00:00<00:01, 61.78frames/s]
Labeling video:  57%|█████████████████████████████████████████████████████▉                                        | 74/129 [00:01<00:01, 51.93frames/s]
Labeling video:  62%|██████████████████████████████████████████████████████████▎                                   | 80/129 [00:01<00:01, 48.75frames/s]
Labeling video:  67%|██████████████████████████████████████████████████████████████▋                               | 86/129 [00:01<00:00, 46.60frames/s]
Labeling video:  71%|██████████████████████████████████████████████████████████████████▎                           | 91/129 [00:01<00:00, 44.76frames/s]
Labeling video:  74%|█████████████████████████████████████████████████████████████████████▉                        | 96/129 [00:01<00:00, 42.79frames/s]
Labeling video:  78%|████████████████████████████████████████████████████████████████████████▊                    | 101/129 [00:01<00:00, 41.00frames/s]
Labeling video:  82%|████████████████████████████████████████████████████████████████████████████▍                | 106/129 [00:01<00:00, 39.25frames/s]
Labeling video:  85%|███████████████████████████████████████████████████████████████████████████████▎             | 110/129 [00:02<00:00, 37.79frames/s]
Labeling video:  88%|██████████████████████████████████████████████████████████████████████████████████▏          | 114/129 [00:02<00:00, 36.63frames/s]
Labeling video:  91%|█████████████████████████████████████████████████████████████████████████████████████        | 118/129 [00:02<00:00, 35.40frames/s]
Labeling video:  95%|███████████████████████████████████████████████████████████████████████████████████████▉     | 122/129 [00:02<00:00, 33.83frames/s]
Labeling video: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 48.52frames/s]
Video labeled at 25.0fps saved to /Users/jaehyunshin/Desktop/envision/D_21/output/labeled_hair_salon.mp4
Generating ELAN file...
Done processing hair_salon.mp4 with LIGHTGBM
```python
import pandas as pd
import os
import glob as glob
# lets list the output
outputfiles = glob.glob(outputfolder + '/*')
for file in outputfiles:
    print(os.path.basename(file))

# load one of the predictions
csvfilessegments = glob.glob(outputfolder + '/*segments.csv')
df = pd.read_csv(csvfilessegments[0])
df.head()
```
concert_hall.mp4_features.npy
darkroom.mp4_predictions.csv
labeled_darkroom.mp4
vicar.mp4_segments.csv
haircut.mp4_segments.csv
concert_hall.mp4.eaf
police_officer.mp4_segments.csv
restaurant.mp4_segments.csv
frying_pan.mp4_predictions.csv
labeled_bible.mp4
preach.mp4_segments.csv
labeled_photographer.mp4
labeled_chef.mp4
labeled_prison.mp4
darkroom.mp4_segments.csv
labeled_concert_hall.mp4
bible.mp4_predictions.csv
singer.mp4_features.npy
photographer.mp4_segments.csv
labeled_haircut.mp4
microphone.mp4_segments.csv
vicar.mp4.eaf
bible.mp4_features.npy
camera.mp4_segments.csv
labeled_take_photo.mp4
restaurant.mp4_predictions.csv
scissors.mp4_segments.csv
labeled_vicar.mp4
cook.mp4_segments.csv
hairdresser.mp4_segments.csv
haircut.mp4.eaf
frying_pan.mp4_features.npy
hair_salon.mp4_features.npy
handcuffs.mp4_predictions.csv
hairdresser.mp4_predictions.csv
labeled_sing.mp4
labeled_camera.mp4
church.mp4_features.npy
prison.mp4_segments.csv
sing.mp4_features.npy
take_photo.mp4_segments.csv
singer.mp4_predictions.csv
chef.mp4.eaf
labeled_hair_salon.mp4
handcuffs.mp4.eaf
handcuffs.mp4_segments.csv
church.mp4.eaf
chef.mp4_features.npy
take_photo.mp4.eaf
arrest.mp4_features.npy
police_officer.mp4_predictions.csv
arrest.mp4_predictions.csv
take_photo.mp4_predictions.csv
cook.mp4_predictions.csv
restaurant.mp4.eaf
preach.mp4_features.npy
labeled_preach.mp4
photographer.mp4.eaf
restaurant.mp4_features.npy
police_officer.mp4_features.npy
hairdresser.mp4.eaf
labeled_arrest.mp4
haircut.mp4_features.npy
concert_hall.mp4_predictions.csv
vicar.mp4_features.npy
police_officer.mp4.eaf
concert_hall.mp4_segments.csv
microphone.mp4_features.npy
photographer.mp4_features.npy
microphone.mp4.eaf
singer.mp4.eaf
labeled_hairdresser.mp4
singer.mp4_segments.csv
bible.mp4.eaf
darkroom.mp4_features.npy
arrest.mp4.eaf
sing.mp4_predictions.csv
haircut.mp4_predictions.csv
vicar.mp4_predictions.csv
darkroom.mp4.eaf
hair_salon.mp4_segments.csv
frying_pan.mp4_segments.csv
hairdresser.mp4_features.npy
cook.mp4_features.npy
camera.mp4_predictions.csv
cook.mp4.eaf
scissors.mp4_features.npy
labeled_handcuffs.mp4
labeled_cook.mp4
labeled_frying_pan.mp4
camera.mp4_features.npy
prison.mp4.eaf
bible.mp4_segments.csv
labeled_church.mp4
labeled_police_officer.mp4
scissors.mp4.eaf
labeled_restaurant.mp4
camera.mp4.eaf
preach.mp4.eaf
microphone.mp4_predictions.csv
prison.mp4_predictions.csv
hair_salon.mp4_predictions.csv
labeled_scissors.mp4
arrest.mp4_segments.csv
hair_salon.mp4.eaf
preach.mp4_predictions.csv
frying_pan.mp4.eaf
chef.mp4_segments.csv
handcuffs.mp4_features.npy
labeled_microphone.mp4
church.mp4_predictions.csv
labeled_singer.mp4
take_photo.mp4_features.npy
chef.mp4_predictions.csv
sing.mp4_segments.csv
prison.mp4_features.npy
church.mp4_segments.csv
photographer.mp4_predictions.csv
sing.mp4.eaf
scissors.mp4_predictions.csv
|start_time|end_time|labeled|label|duration|
|----------|--------|-------|-----|--------|
|0     |4.33333|3.766667|1|Gesture|3.333333|

```python
from moviepy.editor import VideoFileClip
import os
videoslabeled = glob.glob(os.path.join(outputfolder, '*labeled*.mp4'))

videoname="arrest"

# need to rerender
clip = VideoFileClip(videoslabeled[0])
clip.write_videofile(os.path.join(outputfolder, f'labeled_{videoname}.mp4'))
Video(os.path.join(outputfolder, f'labeled_{videoname}.mp4'), embed=True, width=300)
```
```python
# Step 2: Cut videos into segments
from envisionhgdetector import utils
segments = utils.cut_video_by_segments(outputfolder)
```
Labeling video:  54%|██████████████████████████████████████████████████▎                                           | 61/114 [13:46<00:00, 72.31frames/s] \
Moviepy - Building video /Users/jaehyunshin/Desktop/envision/D_21/output/gesture_segments/vicar.mp4/vicar.mp4_segment_1_Gesture_0.43_3.77.mp4. \
Moviepy - Writing video /Users/jaehyunshin/Desktop/envision/D_21/output/gesture_segments/vicar.mp4/vicar.mp4_segment_1_Gesture_0.43_3.77.mp4 \


t:   0%|                                                                                                               | 0/84 [00:00<?, ?it/s, now=None] \
t:  95%|████████████████████████████████████████████████████████████████████████████████████████████████▏    | 80/84 [00:00<00:00, 791.20it/s, now=None] \
Labeling video:  54%|██████████████████████████████████████████████████▎                                           | 61/114 [13:47<00:00, 72.31frames/s] \
Moviepy - Done ! \
Moviepy - video ready /Users/jaehyunshin/Desktop/envision/D_21/output/gesture_segments/vicar.mp4/vicar.mp4_segment_1_Gesture_0.43_3.77.mp4 \
Created segment and features: vicar.mp4_segment_1_Gesture_0.43_3.77.mp4 \
Completed processing segments for vicar.mp4 \
Moviepy - Building video /Users/jaehyunshin/Desktop/envision/D_21/output/gesture_segments/haircut.mp4/haircut.mp4_segment_1_Gesture_0.63_5.10.mp4. \
Moviepy - Writing video /Users/jaehyunshin/Desktop/envision/D_21/output/gesture_segments/haircut.mp4/haircut.mp4_segment_1_Gesture_0.63_5.10.mp4 

```python
# Step 3: Create paths
gesture_segments_folder = os.path.join(outputfolder, "gesture_segments")
retracked_folder = os.path.join(outputfolder, "retracked")
analysis_folder = os.path.join(outputfolder, "analysis")

print(f"\nLooking for segments in: {gesture_segments_folder}")
if os.path.exists(gesture_segments_folder):
    segment_files = [f for f in os.listdir(gesture_segments_folder) if f.endswith('.mp4')]
    print(f"Found {len(segment_files)} segment files")
else:
    print("Gesture segments folder not found!")

# Step 3: Retrack gestures with world landmarks
print("\nStep 4: Retracking gestures...")
tracking_results = detector.retrack_gestures(
    input_folder=gesture_segments_folder,
    output_folder=retracked_folder
)
print(f"Tracking results: {tracking_results}")
```

```python
# Videos list
import glob

videoslabeled = glob.glob(os.path.join(outputfolder, 'retracked', 'tracked_videos', '*.mp4'))
videoslabeled = sorted(videoslabeled, key=lambda x: os.path.basename(x).lower())

print("Available labeled videos:")
for idx, path in enumerate(videoslabeled):
    print(f"{idx}: {os.path.basename(path)}")
```
Available labeled videos:
0: arrest.mp4_segment_1_Gesture_0.40_3.00_tracked.mp4 \
1: arrest.mp4_segment_2_Gesture_3.60_4.27_tracked.mp4 \
2: arrest.mp4_segment_3_Gesture_4.60_5.20_tracked.mp4 \
3: bible.mp4_segment_1_Gesture_0.17_4.30_tracked.mp4 \
4: camera.mp4_segment_1_Gesture_0.23_4.70_tracked.mp4 \
5: chef.mp4_segment_1_Gesture_0.60_5.27_tracked.mp4 \
6: church.mp4_segment_1_Gesture_0.63_5.97_tracked.mp4 \
7: concert_hall.mp4_segment_1_Gesture_0.80_4.70_tracked.mp4 \
8: cook.mp4_segment_1_Gesture_0.13_2.17_tracked.mp4 \
9: cook.mp4_segment_2_Gesture_2.60_4.40_tracked.mp4 \
10: darkroom.mp4_segment_1_Gesture_1.07_4.37_tracked.mp4 \
11: frying_pan.mp4_segment_1_Gesture_0.13_4.17_tracked.mp4 \
12: hair_salon.mp4_segment_1_Gesture_0.37_4.70_tracked.mp4 \
13: haircut.mp4_segment_1_Gesture_0.63_5.10_tracked.mp4 \
14: hairdresser.mp4_segment_1_Gesture_0.70_3.43_tracked.mp4 \
15: handcuffs.mp4_segment_1_Gesture_0.13_1.87_tracked.mp4 \
16: microphone.mp4_segment_1_Gesture_0.63_3.47_tracked.mp4 \
17: photographer.mp4_segment_1_Gesture_0.50_4.93_tracked.mp4 \
18: police_officer.mp4_segment_1_Gesture_0.27_4.77_tracked.mp4 \
19: preach.mp4_segment_1_Gesture_0.37_3.47_tracked.mp4 \
20: prison.mp4_segment_1_Gesture_0.13_3.77_tracked.mp4 \
21: restaurant.mp4_segment_1_Gesture_0.53_3.67_tracked.mp4 \
22: scissors.mp4_segment_1_Gesture_0.93_1.63_tracked.mp4 \
23: scissors.mp4_segment_2_Gesture_1.97_3.37_tracked.mp4 \
24: sing.mp4_segment_1_Gesture_1.03_3.53_tracked.mp4 \
25: singer.mp4_segment_1_Gesture_0.47_1.90_tracked.mp4 \
26: take_photo.mp4_segment_1_Gesture_0.17_2.90_tracked.mp4 \
27: vicar.mp4_segment_1_Gesture_0.43_3.77_tracked.mp4

```python
# lets show the new tracked videos and sorted

videoslabeled = glob.glob(os.path.join(outputfolder, 'retracked', 'tracked_videos', '*.mp4'))
videoslabeled = sorted(videoslabeled, key=lambda x: os.path.basename(x).lower())  # 파일명 기준 정렬

# need to rerender
clip = VideoFileClip(videoslabeled[0])
clip.write_videofile("./retracked.mp4")
Video("./retracked.mp4", embed=True, width=400)
```
Labeling video:  54%|██████████████████████████████████████████████████▎                                           | 61/114 [29:04<00:00, 72.31frames/s] \
Moviepy - Building video ./retracked.mp4. \
Moviepy - Writing video ./retracked.mp4 \


t:   0%|                                                                                                               | 0/65 [00:00<?, ?it/s, now=None] \
Labeling video:  54%|██████████████████████████████████████████████████▎                                           | 61/114 [29:04<00:00, 72.31frames/s] \
Moviepy - Done ! \
Moviepy - video ready ./retracked.mp4 \
videos
```python
import pandas as pd
import os
# lets list the output
outputfiles = glob.glob(os.path.join(outputfolder, 'analysis', '*'))
for file in outputfiles:
    print(os.path.basename(file))

# load one of the predictions
csvfilessegments = glob.glob(os.path.join(outputfolder, 'analysis', '*kinematic_features.csv'))
df = pd.read_csv(csvfilessegments[0])
df.head()
```
dtw_distances.csv
kinematic_features.csv
gesture_visualization.csv



2. Evaluation von Envisiohgdetector
  - [ ] Gesture, Non-Gesture Labeling - Durch ELAN
  - [ ] Vergleichen mit durch einvisiohgdetector hergestellte **prediction.csv**
  - [ ] Evaluieren

***

### Nächster Schritt: Mit vLLM testen
