import numpy as np
from scipy.signal import argrelmax
from scipy.spatial.distance import cosine


def boundary_prediction(embedded_seqFfromP, embedded_seqPfromF, order=5):
    """Predicts the event boundaries at time t given
    the past and future visual context.
    Inputs:
        visual context given the past
        visual context given the future
        order=size of the window for the local maxima  
    Output:
        prediction vector (1/0)
    """
    # at time t, we want the difference between context from the past at t-1
    # and context from the future at t+1:
    signal = np.array([cosine(a, b) for 
                       (a, b) in zip(embedded_seqFfromP[:-2],
                                     embedded_seqPfromF[2:])])

    # First and last context predictions will be noisy, better to ignore:
    signal[0] = 0
    signal[-1] = 0

    local_max = argrelmax(signal, order=order)[0]
    th = np.mean(signal[local_max])
    prediction = np.zeros(len(signal))
    prediction[local_max[signal[local_max] > th]] = 1

    # First and last frames are always event boundaries 
    # (context prediction goes from t=1 to t=N-1):
    return np.hstack((1, prediction, 1))


def get_visual_context(model, params, features):
    """Predict the visual context at each timestep.
    Inputs:
        model
        params
        features [user, day, frame, gt, visual_feat]
    Output:
        [user, day, frame, gt]
        visual context given the past
        visual context given the future
    """
    sequencePtoF = np.array([features[:-1]])
    sequenceFtoP = np.array([features[::-1][:-1]])

    """ Future: """
    embedded_seqF = model.embed_sequence(sequencePtoF)[0]
    embedded_seqFfromP = np.pad(embedded_seqF, ((0, 1), (0, 0)), 'reflect')
    """ Past: """
    embedded_seqP = model.embed_sequence(sequenceFtoP)[0]
    embedded_seqPfromF = np.pad(embedded_seqP,
                                ((0, 1), (0, 0)), 'reflect')[::-1]

    return embedded_seqFfromP, embedded_seqPfromF


class evaluation_measures(object):
    def __init__(self, TP=0, FP=0, TN=0, FN=0, y_pred=None, y_gt=None, win=5):
        self.reset(TP=TP, FP=FP, TN=TN, FN=FN,
                   y_pred=y_pred, y_gt=y_gt, win=win)

    def fix_boundaries(self, x):
        return x[np.hstack((True, x[:-1] + 1 != x[1:]))]

    def reset(self, TP=0, FP=0, TN=0, FN=0, y_pred=None, y_gt=None, win=None):
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.y_pred = y_pred
        self.y_gt = y_gt
        if win is not None:
            self.win = win
        self.recall = None
        self.precision = None

    def scores(self, y_pred=None, y_gt=None, total_frames=None):
        if y_pred is not None and y_gt is not None:
            self.reset(y_pred=y_pred, y_gt=y_gt)
        else:
            y_gt = self.y_gt
            y_gt[-1] = 1
            y_pred = self.y_pred
        """validate input:"""
        if len(y_pred) == len(y_gt) and max(y_pred) < 2 and max(y_gt) < 2 and \
                np.sum(y_pred == 0) > 1 and np.sum(y_gt == 0) > 1:
            p = np.where(y_pred)[0]
            gt = np.where(y_gt)[0]
            total_frames = len(y_gt)
        elif total_frames is not None:
            p = y_pred
            gt = y_gt
        else:
            raise ValueError('Your inputs are not of the right kind!')

        gt = self.fix_boundaries(gt)
        Neg = total_frames - len(gt)

        """compute scores:"""
        for el_p in p:
            aux = np.where(np.abs(gt-el_p) <= self.win)[0]
            if len(aux):
                self.TP += 1
                gt = np.delete(gt, aux[0])
            else:
                self.FP += 1
        self.FN = len(gt)
        self.TN = Neg - self.FP

    def get_recall(self, y_pred=None, y_gt=None):
        if y_pred is not None and y_gt is not None:
            self.reset(y_pred=y_pred, y_gt=y_gt)

        if (self.TP + self.FP + self.TN + self.FN) == 0:
            self.scores()

        self.recall = 1.*self.TP/max(1, self.TP+self.FN)
        return self.recall

    def get_precision(self, y_pred=None, y_gt=None):
        if y_pred is not None and y_gt is not None:
            self.reset(y_pred=y_pred, y_gt=y_gt)

        if (self.TP + self.FP + self.TN + self.FN) == 0:
            self.scores()

        self.precision = 1.*self.TP/max(1, self.TP+self.FP)
        return self.precision

    def f_measure(self, y_pred=None, y_gt=None):
        if y_pred is not None and y_gt is not None:
            self.reset(y_pred=y_pred, y_gt=y_gt)

        if self.recall is None and self.precision is None:
            self.get_precision()
            self.get_recall()
        return 2.*self.precision*self.recall / max(1, self.recall +
                                                   self.precision)