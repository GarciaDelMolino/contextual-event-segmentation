import numpy as np
import pandas as pd


def dataset_structure(path,
                      dataset_filename='TestDB.npz',
                      gt_path=None,
                      users=None):
    """Builds a dictionary with the lifelog structure.
    Input:
        path to dataset
        filename where the dictionary will be stored
        path to the GT (if available)
        id of the users to include
    Returns:
        dictionary with structure
        dataset = {'path': path,
                   'users': {user_id: {'id': 'folder_name',
                                       'days': {day_id: day_dict}}}}
              where day_dict = {'date': 'folder_name',
                                'GT': {image_id: image_name.jpg},
                                'images': {image_id: image_name.jpg}}
    """
    def find_gt_from_file(pathGT, filename, im):
        xlsx = False
        xl = pd.ExcelFile(pathGT + filename + '.xls')
        df = xl.parse(xl.sheet_names[0])
        annotations = df.keys()[1:]
        # only take 1st annotation. ignore others:
        b = [f+'.jpg' for b in df[annotations[0]]
             for f in str.split(str(b)) if f != 'nan']
        gt = (np.isin(im, b))
        gt[0] = 1
        gt[-1] = 1
        return gt.astype(int)

    try:
        dataset = np.load(path + dataset_filename)['dataset'].item()
    except IOError as e:
        print e
        if users is None:
            dataset = {'path': path[:-1],
                       'users': {0: {'id': '',
                                     'days': {0: {'date': ''}}}}}
        else:
            raise('WARNING: Modify this code four your DB folder structure!')
            dataset = {'users': {i: {'id': 'user' + str(i)} for i in users}}
            for u, user in dataset['users'].items():
                days = {i: {'date': day} for i, day in
                        enumerate(os.listdir(dataset['path'] + user['id']))}
                dataset['users'][u]['days'] = days

        for u, user in dataset['users'].items():
            for d, day in dataset['users'][u]['days'].items():
                frames = dataset['users'][u]['days'][d]
                day_path = dataset['path'] + user['id'] + '/' + day['date']
                images = [im for im in os.listdir(day_path) if
                          im.endswith(".jpg") and not
                          im.startswith(".") and not
                          im.endswith("(1).jpg")]
                im = {i: im for i, im in enumerate(sorted(images))}
                frames['images'] = im
                if gt_path is not None:
                    frames['GT'] = {i: gt for i, gt in
                                    zip(im.keys(),
                                        find_gt_from_file(gt_path,
                                                          frames['date'],
                                                          im.values()))}
        np.savez(path + dataset_filename, dataset=dataset)
    return dataset


def extract_data_DB(dataset, rotate=None, return_feat=False):
    """ Extracts features for testing given a dictionary.
    Parameters
    ----------
    dataset dictionary; format: dataset['users'][u]['days'][d]['images'][i]

    Yields
    -------
    u
    d
    i
    gt (zeros if not available)
    descriptors of images "dataset['users'][u]['days'][d]['images']"
    """
    VF = VF_extractor()
    for u, user in dataset['users'].items():
        for d, day in user['days'].items():
            gt = []
            frames = []
            features = []
            print('extracting for day %d' % d)
            path = dataset['path'] + user['id'] + '/' + day['date']
            path += ('' if (path[-1] == '/') else '/')
            try:
                features = np.load(path + ('featuresDay%d.npz' % d)
                                   )['features'].item()
            except IOError as e:
                print e
                for i in tqdm(sorted(day['images'].keys())):
                    im_path = path + day['images'][i]
                    try:
                        feat = VF.get_feat(im_path, rotate=rotate)
                        frames.append(i)
                        features.append(feat)
                        if 'GT' in day.keys():
                            gt.append(day['GT'][i])
                        else:
                            gt.append(0)
                    except Exception as e:
                        print e, im_path
                features = np.hstack((np.vstack((u*np.ones(len(frames)),
                                                 d*np.ones(len(frames)),
                                                 frames,
                                                 gt)).T,
                                      features))
                np.savez(path + ('featuresDay%d.npz' % d), features=features)
            if return_feat:
                yield features


class VF_extractor(self):
    """Class to extract visual features from InceptionV3 """
    def __init__(self, include_top=False, pooling='max',
                 target_size=(299, 299)):
        """Default inputs: 
            include_top=False,
            pooling='max',
            target_size=(299, 299)
        """
        from keras.preprocessing import image
        from keras.applications.inception_v3 import InceptionV3,
        from keras.applications.inception_v3 import preprocess_input
        self.model = InceptionV3(include_top=include_top, pooling=pooling)
        self.target_size = target_size

    def get_feat(self, im_path, rotate=None):
        """Returns the visual feature given an image path. 
        Optional input: rotate = 1 for clockwise/ -1 for counterclockwise"""
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        if rotate is not None:
            x = np.swapaxes(x[::-rotate], 0, 1)[::rotate]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return self.model.predict(x)


def get_svm_data(shots, desc):
    """Given the location of the shots, and the visual features,
    returns the datapoints needed for the SVM  cluster analysis.
    Inputs:
        Boundary locations
        Visual descriptor of the lifelog
    Returns:
        betaCV,
        norm_cut,
        cluster cross-correlation,
        calinski_harabaz_score,
        compactness joined clusters,
        compactness cluster before boundaries,
        compactness cluster after boundaries
    """
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
