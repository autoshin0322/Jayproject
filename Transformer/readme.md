## Dataset:
**GESRes Dataset:1Plitician, 2Clinician, 3Educator** \
**MULTISIMO Dataset: P\*\*_S\*\*_front-video_Z_S_L, S\*\*_all_video_Z_S_L** \
**SaGA Dataset: V\*\*** \
**ZHUBO Dataset: 9-\*\*\*.h264** \
**Va.Si.Li-lab Dataset: video\*\*** \


## Getting ready:

Steps Dataset to prepare
1. To train posevit need a pair of *.features.npy (keypoints), *.labels.csv.
2. For make features.npy use envisionhgdetector
3. envisionhgdetector makes *.mp4_features.npy
4. .npy files in ~/Transformer/data/features, .csv files in ~/Transformer/data/labels

additionally
5. for new data (not trained) followed the Steps
6. .npy files in ~/Transformer/data/TestFeatures, .csv files in ~/Transformer/data/TestLabels

## How to Use:

work in location " Jayproject/Transformer"

1. python make_index.py
	- make index.csv for all dataset and save in Transformer/data
	- name, features_path, labels_path
2. python train_eval.py
	- split index_all to index_train.csv, index_val.csv, index_test.csv and save in Transformer/data
	- train the model and save checkpoints in ouputs/ckpts


(for additional Step 5, 6)
3. python posevit/infer.py
	- use checkpoint, that made in Step 2
	- make predictions and save in outputs/preds

4. python posevit/BinaryEval.py
	- evaluation with the predictions from step 3 and ground_truth(~/Transformer/data/TestLabels)
	 and results saved in ouputs/reports (CM.png and eval_summary.csv)

