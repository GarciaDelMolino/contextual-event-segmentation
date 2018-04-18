"""
Created on Fri Apr 13 00:00:00 2018

@author: ana
If you use this code, please cite our paper 
"Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams"
"""
from extraction_utils import dataset_structure, extract_data_DB
from testing_utils import get_visual_context, boundary_prediction
from testing_utils import evaluation_measures
from models import init_model
from keras import backend as K
import numpy as np
PATH_VCP = '/home/ana/models/VCP'  # '/home/ana/models/TruePred_TrueCond_1024_RMSprop0.0005_0_d33b3'
DATA_PATH = 'test_data/'
#DATA_PATH = '/data/EDUB-Seg/images/'
#IMAGGA_Set = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 3), (4, 0),
#              (5, 0), (5, 1), (5, 2), (7, 0), (7, 1)]
# paths = [DATA_PATH + 'Subject%d_Set%d/' % (u, d+1) for (u, d) in IMAGGA_Set]
K.clear_session()
model, params = init_model(PATH_VCP)

all_scores = {}
scores = evaluation_measures(win=5)
path = DATA_PATH
""" Extract features for the lifelog: """
dataset = dataset_structure(path,
                            dataset_filename='TestDB.npz',
                            gt_path=path + 'GTfile',
                            users=None)
features = next(extract_data_DB(dataset, return_feat=True))

""" Predict visual context and presence of boundaries: """
info = features[:, :4]
embedded_seqFfromP, embedded_seqPfromF = get_visual_context(model,
                                                            params,
                                                            features[:, 4:])
prediction = boundary_prediction(embedded_seqFfromP,
                                 embedded_seqPfromF,
                                 order=5)

""" Evaluate the performance: """
scores.reset(y_pred=prediction, y_gt=info[:, 3])
print 'performance in terms of FM, P, R:',
print scores.f_measure(), scores.get_precision(), scores.get_recall()

