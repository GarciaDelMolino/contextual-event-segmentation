# Contextual Event Segmentation

This repository includes the functions needed to run Contextual Event Segmentation as presented in our paper ["Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams."](https://garciadelmolino.github.io/ces.html)

## What is Contextual Event Segmentation (CES)? Why is it useful?
Given a continuous stream of photos, we, as humans, would identify the start of an event if the new frame differs from the expectation we have generated. The proposed model is analogous to such intuitive framework of perceptual reasoning. CES consists of two modules:

1. the Visual Context Predictor (VCP), an LSTM network that predicts the visual context of the upcoming frame, either in the past or in the future depending on the sequence ordering. An auto-encoder architecture is used to train VCP with the objective of reaching minimum prediction mse.
2. the event boundary detector, that compares the visual context at each time-step given the frame sequence from the past, with the visual context given the sequence in the future.

#### CES in action for one example lifelog from EDUB-Seg:
![example users in our dataset](https://garciadelmolino.github.io/files/CES-qualitative.png)

CES is able to ignore occasional occlusions as long as the different points of view span less frames than CES’ memory span (A). It is also capable of detecting boundaries that separate heterogeneous events such as riding a bike on the street and shopping at the supermarket (C, D). Most of the boundaries not detected by CES correspond to events that take place within the same physical space (B) and short transitions (C, D), e.g. parking the bike.


## Steps to reproduce:
1. Clone this repo, create a `test_data` folder within it, and extract [this test lifelog and GT from EDUB-Seg](https://drive.google.com/open?id=1vBDdLR1IUXOSMB2p1gUlxpB5bJVW0fvE) to it.
2. Download the model architecture and weights [here](https://data.mendeley.com/datasets/ktps5my69g/1)
3. Change `PATH_VCP` in `demo.py` to match the location of your downloaded model architecture and weights.
4. Run `demo.py`

The dataset used to train the model, as well as the model weights, can be found [here](https://data.mendeley.com/datasets/ktps5my69g/1).


### How to execute CES on your own lifelog(s):
If you want to execute CES on your own lifelogs (the images and ground truth, if available), just follow the instructions from [this Wiki page](https://github.com/GarciaDelMolino/CES/wiki/How-to-execute-CES-on-your-own-lifelog(s))

### How to use VCP for you own application:
The Visual Context Predictor can be used for many applications, such as retrieval, activity detection from low time resolution videos, and summarization. Pointers to how to re-train it for your own data can be found in [this Wiki page](https://github.com/GarciaDelMolino/CES/wiki/How-to-use-VCP-for-you-own-application)



## What's in this repo?

The repository is organized as follows:

- `demo.py`: full demo pipeline to test one sample data.
- `__init__.py`: `class VCP` to load, train and test the Visual Context Prediction model, with `init_model` and `params_VCP` definition; `prunning_SVM` model; and training callback `EarlyStoppingTH`.
- `extraction_utils.py`: method `VF_extractor` to extract the visual features from `InceptionV3`; functions to create the dataset from a folder of images and extract the visual features;  functions to extract training and testing data for the `prunning_SVM` model.
- `testing_utils.py`: functions to extract the visual context from the testing data, find the event boundaries, and evaluate the event segmentation.


## Citation:
If you found this code or the R3 dataset useful, please cite the following publication:

    @inproceedings{garcia2018predicting,
      title={{Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams}},
      author={Garcia del Molino, Ana and Lim, Joo-Hwee and Tan, Ah-Hwee},
      booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
      pages={10--17},
      year={2018},
      organization={ACM}
    }
