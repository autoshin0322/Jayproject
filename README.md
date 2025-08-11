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

### Aktueller Stand
1. envisionhgdetector testen
  - [x] TTLab Dataset
  - [x] University of Edinburgh Dataset

'''python
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
'''

2. Evaluation von Envisiohgdetector
  - [ ] Gesture, Non-Gesture Labeling - Durch ELAN
  - [ ] Vergleichen mit durch einvisiohgdetector hergestellte **prediction.csv**
  - [ ] Evaluieren

***

### Nächster Schritt: Mit vLLM testen
