Externe Datasets:
ZHUBO Dataset
Vasililab Dataset
SaGA Dataset
MULTISIMO Dataset
GESRes Dataset
M3D TED Dataset

paths in current folder:
-labels
-testlabels

-features
-testfeatures

1.labels and features are just for train
2.testlabels and testfeatures are just for test

*** We have 3 different type of test ***

1. Nur externe Daten auf Vasililab: Alle externe Daten befinden sich in labels und features.
Vasililab Daten sind in testlabels und testfeatures

2. Jeweils eins ausgelassen der externen auf Vasililab: Bleib die Vasililab Daten noch in testlabels und testfeatures
Jeweilige Test sind ein Dataset ausgelassen.
Also in labels und features bleiben beim jeweiligen Test nur 4 Datasets.

3. Teil von Vasililab + externe auf teil von Vasililab: Alle externe Datasets und
Teil von Vasililab in labels und features.
Der Rest von Vasililab in testlabels und testfeatures.
