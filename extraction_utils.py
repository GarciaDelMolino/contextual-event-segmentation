import numpy as np


def get_svm_data(shots, desc):
    from scipy.spatial.distance import cdist
    from sklearn.metrics import calinski_harabaz_score
    margin = 15

    shotsA = (np.vstack((np.maximum(shots - margin, 0),
                         shots)).T).reshape([-1, 1])[:, 0]
    shotsB = (np.vstack((shots,
                         np.minimum(shots + margin, len(desc)))).T
              ).reshape([-1, 1])[:, 0]

    def median_to_min_correlation():
        dist = np.array([np.median(np.min(cdist(desc[a[0]:a[1]],
                                                desc[b[0]:b[1]],
                                                'correlation'),
                                          axis=0))
                         for a, b in zip(shotsA,
                                         shotsB)])
        return dist

    def get_compactness_sc():
        def measure(points):
            return np.sum(cdist(np.array([np.mean(points, axis=0)]),
                                points, 'cosine'))
        c = np.array([measure(desc[a[0]:a[1]]) for
                      a in zip(shotsA,
                               shotsB)])
        c0 = c[:-1:2]  # compactness previous event
        c1 = c[1::2]  # compactness next event
        c = np.array([measure(desc[a[0]:a[1]]) for
                      a in zip(shotsA[:-1:2],
                               shotsB[1::2])])  # compactness union of events
        return c, c0, c1

    def betaCV_normCut():
        def measure(points):
            return np.sum(cdist(np.array([np.mean(points, axis=0)]),
                                points, 'sqeuclidean')/1250.)
        aux = [(desc[a[0]:a[1]]) for a in zip(shotsA, shotsB)]
        aux = np.array([[2*len(a)*measure(a), 2*len(b)*measure(b),
                         2*measure(a, b),
                         len(a), len(b)] for
                        (a, b) in zip(aux[:-1:2], aux[1::2])])
        betaCV = ((aux[:, 0] + aux[:, 1]) / (
                    aux[:, 3] * (aux[:, 3] - 1) + aux[:, 4] * (aux[:, 4] - 1))
                  ) / (aux[:, 2] / (aux[:, 3] * aux[:, 4]))
        norm_cut = (aux[:, 2] / (aux[:, 2] + aux[:, 0])) + (
                    aux[:, 2] / (aux[:, 2] + aux[:, 1]))
        return betaCV, norm_cut

    def get_CH():
        c = [(desc[a[0]:a[1]]) for a in zip(shotsA, shotsB)]
        ch = [calinski_harabaz_score(np.vstack((a, b)),
                                     np.hstack((np.zeros(len(a)),
                                                np.ones(len(b))))) for
              (a, b) in zip(c[:-1:2], c[1::2])]
        return np.array(ch)

    d1 = median_to_min_correlation()
    ch = get_CH()
    (c, c0, c1) = get_compactness_sc()
    (betaCV, norm_cut) = betaCV_normCut()

    return np.array([betaCV, norm_cut, d1, ch, c, c0, c1]).T
