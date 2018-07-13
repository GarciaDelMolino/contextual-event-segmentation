# CES

This repository includes the functions needed to run Contextual Event Segmentation as presented in our paper "Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams."

The repository is organized as follows:
- `demo.py`: full demo pipeline to test one sample data.
- `__init__.py`: `class VCP` to load, train and test the model, with `init_model` and `params_VCP` definition; `prunning_SVM` model; and training callback `EarlyStoppingTH`.
- `extraction_utils.py`: method `VF_extractor` to extract the visual features from `InceptionV3`; functions to create the dataset from a folder of images and extract the visual features;  functions to extract training and testing data for the `prunning_SVM` model.
- `testing_utils.py`: functions to extract the visual context from the testing data, find the event boundaries, and evaluate the event segmentation.


The data used to train the model, as well as the model weights, can be found at http://dx.doi.org/10.17632/ktps5my69g.1

## Steps to reproduce:
1. Clone this repo, create a `test_data` folder within it, and extract [this test lifelog and GT from EDUB-Seg](https://drive.google.com/open?id=1vBDdLR1IUXOSMB2p1gUlxpB5bJVW0fvE) to it.
2. Download the model architecture and weights [here](http://dx.doi.org/10.17632/ktps5my69g.1)
3. Change `PATH_VCP` in L. 15 of `demo.py` to match the location of your downloaded model architecture and weights.
4. Run `demo.py`

### How to execute CES on your own lifelog(s):
If you want to execute CES on your own lifelogs (the images and ground truth, if available), make sure you modify L.47ff of `extraction_utils.py` to match the names of your data folder structure. 

1. Extract the visual features for your lifelog (if the ground Truth is not available, set `gt_path=None`).:
```
from extraction_utils import dataset_structure, extract_data_DB
dataset = dataset_structure("path_to_your_lifelogs",
                            dataset_filename='TestDB.npz',
                            gt_path=path + 'GTfile')
features = next(extract_data_DB(dataset, return_feat=True))
```

2. Load the VCP model:
```
from models import init_model
K.clear_session()
model, params = init_model("path_to_VCP_without_extension")  % eg. "models/VCP"
```

3. Get the visual context:
```
from testing_utils import get_visual_context
embedded_seqFfromP, embedded_seqPfromF = get_visual_context(model,
                                                            params,
                                                            features[:, 4:]
                                                            )
```

4. Find local maximas, and filter the event boundaries:
```
from testing_utils import boundary_prediction
prediction = boundary_prediction(embedded_seqFfromP,
                                 embedded_seqPfromF,
                                 order=5)
```

5. If ground truth is available, you can evaluate the performance:
```
import numpy as np
from testing_utils import evaluation_measures
scores = evaluation_measures(y_pred=prediction, y_gt=features[:, 3], win=5)
print scores.f_measure(), scores.get_precision(), scores.get_recall()])
```

## Usage of VCP for you own application:
- Load the trained model:
```
from models import init_model
% Download the model architecture and weights from http://dx.doi.org/10.17632/ktps5my69g.1
K.clear_session()
model, params = init_model("path_to_VCP_without_extension")  % eg. "models/VCP"
```
- Train you own VCP model:



