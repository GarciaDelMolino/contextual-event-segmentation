"""
Created on Fri Apr 13 00:00:00 2018

@author: ana
If you use this code, please cite our paper 
"Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams"
"""
from .extraction_utils import dataset_structure, extract_data_DB
from .testing_utils import get_visual_context, boundary_prediction
from .testing_utils import evaluation_measures
from . import init_model
from keras import backend as K
K.clear_session()
PATH_VCP = '/home/ana/models/VCP'  # '/home/ana/models/TruePred_TrueCond_1024_RMSprop0.0005_0_d33b3'
DATA_PATH = 'test_data/'

""" Extract features for the lifelog: """
dataset = dataset_structure(DATA_PATH,
                            dataset_filename='TestDB.npz',
                            gt_path=DATA_PATH + 'GTfile.xls',
                            users=None)
features = next(extract_data_DB(dataset, return_feat=True))

""" Predict visual context and presence of boundaries: """
K.clear_session()
model, params = init_model(PATH_VCP)
info, embedded_seqFfromP, embedded_seqPfromF = get_visual_context(model,
                                                                  params,
                                                                  features)
prediction = boundary_prediction(embedded_seqFfromP,
                                 embedded_seqPfromF,
                                 order=5)

""" Evaluate the performance: """
scores = evaluation_measures(y_pred=prediction, y_gt=info[:, 3])
print 'performance in terms of FM, P, R:',
print scores.f_measure(), scores.get_precision(), scores.get_recall()

