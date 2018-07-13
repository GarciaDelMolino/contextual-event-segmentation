# Contextual Event Segmentation

This repository includes the functions needed to run Contextual Event Segmentation as presented in our paper "Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams."

The repository is organized as follows:

- `demo.py`: full demo pipeline to test one sample data.
- `__init__.py`: `class VCP` to load, train and test the Visual Context Prediction model, with `init_model` and `params_VCP` definition; `prunning_SVM` model; and training callback `EarlyStoppingTH`.
- `extraction_utils.py`: method `VF_extractor` to extract the visual features from `InceptionV3`; functions to create the dataset from a folder of images and extract the visual features;  functions to extract training and testing data for the `prunning_SVM` model.
- `testing_utils.py`: functions to extract the visual context from the testing data, find the event boundaries, and evaluate the event segmentation.


The data used to train the model, as well as the model weights, can be found [here](http://dx.doi.org/10.17632/ktps5my69g.1).

## Citation:
If you found this code or the R3 dataset useful, please cite the following publication:

    Ana García del Molino, Joo-Hwee Lim, and Ah-Hwee Tan. 2018.
    Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams.
    In Proceedings of ACM Multimedia conference (ACMMM’18). ACM, New York, NY, USA.

## Steps to reproduce:
1. Clone this repo, create a `test_data` folder within it, and extract [this test lifelog and GT from EDUB-Seg](https://drive.google.com/open?id=1vBDdLR1IUXOSMB2p1gUlxpB5bJVW0fvE) to it.
2. Download the model architecture and weights [here](http://dx.doi.org/10.17632/ktps5my69g.1)
3. Change `PATH_VCP` in L. 15 of `demo.py` to match the location of your downloaded model architecture and weights.
4. Run `demo.py`

### How to execute CES on your own lifelog(s):
Instructions can be found in [this Wiki page](https://github.com/GarciaDelMolino/CES/wiki/How-to-execute-CES-on-your-own-lifelog(s))

### How to use VCP for you own application:
Instructions can be found in [this Wiki page](https://github.com/GarciaDelMolino/CES/wiki/How-to-use-VCP-for-you-own-application)


