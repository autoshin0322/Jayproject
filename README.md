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
1. **TTLab Goethe-Universität - multiperspektive Videos wenden an Dr. H***** \
2. **University of Edinburgh, Centre for Language Evolution** https://datashare.ed.ac.uk/handle/10283/3191
3. **IFADV** https://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFADVcorpus/
4. **The Multimodal Multidimensional (M3D)** https://osf.io/ankdx/


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

***

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
#### 🥼 Va.Si.Li-Lab
![Image](https://github.com/user-attachments/assets/4b7eb575-6501-46e6-829a-910bd31a1297)

#### 📘 Univercity of Edinburgh
![Image](https://github.com/user-attachments/assets/a9cab613-03e9-41bf-9a1a-90262187e485)
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

Labeling video:   0%|                                                                                                       | 0/129 [00:00<?, ?frames/s] \
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
labeled_bible.mp4....

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

### 🎥 tracked video
#### 🥼 Va.Si.Li-Lab
![Image](https://github.com/user-attachments/assets/38d7b3ea-87f1-4a6e-bd80-0c65a726eff9)

#### 📘 Univercity of Edinburgh
![Image](https://github.com/user-attachments/assets/2e6026d0-1e57-4eae-8493-edefae48fdb6)



2. Evaluation von Envisiohgdetector
  - [ ] Gesture, Non-Gesture Labeling - Durch ELAN
  - [ ] Vergleichen mit durch einvisiohgdetector hergestellte **prediction.csv**
  - [ ] Evaluieren

***

### Nächster Schritt: Mit vLLM testen
