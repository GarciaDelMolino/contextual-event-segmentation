"""
Created on Fri Apr 13 00:00:00 2018

@author: ana
If you use this code, please cite our paper 
"Predicting Visual Context for Unsupervised Event Segmentation in Continuous Photo-streams"
"""
import numpy as np
from keras.layers import Input, Lambda, LSTM
from keras.optimizers import RMSprop
from keras.models import Model, model_from_json
from keras.callbacks import TensorBoard, Callback
from keras import backend as K
import uuid
import h5py
import cPickle


def init_model(model_path, epoch=None):
    try:
        params = np.load(model_path + '.npz')['params'].item()
    except IOError as e:
        print e
        params = params_VCP()

    print 'loading VisualContextPredictor model...'
    model = VCP(params,
                load_filepath=model_path + '_arch.npz')

    print 'loading weights...'
    if epoch is None:
        weights = model_path + '_weights'
    else:
        weights = model_path + ('.%d.hdf5' % epoch)
    model.load_weights(weights)

    print 'Done. Ready to start prediction.'
    return model, params


def params_VCP():
    """set VCP parameters to reasonable defaults"""
    params = {'input_dim': [2048],
              'representation_dim': [1024],
              'timesteps': (10, 10)}

    params['loss'] = 'mean_squared_error'
    params['optimizer'] = 'RMSprop'
    params['batch_size'] = 250
    params['train_dataset'] = 'train_dataset.h5'
    params['val_dataset'] = 'val_dataset.h5'
    params['learning_rate'] = .001
    params['decay'] = .0
    params['log_dir'] = '/tmp/' + str(uuid.uuid4())[-5:]
    return params
    
    
class VCP(object):
    def __init__(self,
                 params=params_VCP(),
                 save_filepath=None,
                 load_filepath=None):
        self.params = params
        if load_filepath is None:
            timesteps = params['timesteps']
            input_dim = params['input_dim']
            encoding_dim = params['representation_dim']
            
            encoding_layers = []
            for n in params['representation_dim'][:-1]:
                encoding_layers.append(LSTM(n))
            encoding_layers.append(LSTM(params['representation_dim'][-1],
                                        return_sequences=True))
            decoding_layers = []            
            for n in params['representation_dim'][:-1][::-1]:
                decoding_layers.append(LSTM(n))
            decoding_layers.append(LSTM(params['input_dim'],
                                        return_sequences=True))
    
            # define model placeholders and architecture:
            self.cond_prediction_model(timesteps, input_dim,
                                       encoding_dim, encoding_layers,
                                       decoding_layers)

            if save_filepath is not None:
                self.save_model(save_filepath, only_arch=True)
        else:
            arch = np.load(load_filepath)['arch'].item()
            self.seq_to_seq = model_from_json(arch['seq_to_seq'])
            self.encoder = model_from_json(arch['encoder'])
            self.predictor = model_from_json(arch['predictor'])

        if params['optimizer'] == 'RMSprop':
                params['optimizer'] = RMSprop(lr=params['learning_rate'],
                                              decay=params['decay'])
        # compile the model:
        self.seq_to_seq.compile(optimizer=params["optimizer"],
                                loss=params["loss"])
        self.graph = None
        self.weight_save_callback = None
        self.other_callbacks = None
        
    def cond_prediction_model(self, timesteps, input_dim,
                              encoding_dim, encoding_layers, decoding_layers):
        inputs = Input(shape=(None, input_dim))  # expected len sum(timesteps)
    
        # encoding layers:
        encoded = encoding_layers[0](inputs)
        for layer in encoding_layers[1:]:
            encoded = layer(encoded)
            # last layer needs to output the sequence

        # this model maps an input to its encoded representation:
        self.encoder = Model(inputs, encoded)            
        # decoding layers:
        decoded = Lambda(lambda encoded:
                         encoded[:, timesteps[0]-1:-1, :])(encoded)
        for layer in decoding_layers:
            decoded = layer(decoded)

        # this model maps an input to its future prediction:
        self.seq_to_seq = Model(inputs, decoded)
        # create a placeholder for an encoded input
        encoded_input = Input(shape=(None, encoding_dim))  # expected len timesteps[1]

        # retrieve the generating layers of the autoencoder model
        decoded_output = decoding_layers[0](encoded_input)
        for layer in decoding_layers[1:]:
            decoded_output = layer(decoded_output)

        # this model predicts a future sequence from the input representation:
        self.predictor = Model(encoded_input, decoded_output)

    def embed_sequence(self, test_data):
        if self.graph is not None:
            with self.graph.as_default():
                encoded_seq = self.encoder.predict(test_data)
        else:
            encoded_seq = self.encoder.predict(test_data)

        return encoded_seq

    def predict_future(self, embedded_data, conditional=False):
        if embedded_data.ndim < 3:
            timesteps = self.params['timesteps'][1]
            embedded_data = np.tile(np.swapaxes(embedded_data[:, :, None],
                                                1, 2),
                                    [1, timesteps, 1])
        if self.graph is not None:
            with self.graph.as_default():
                future = self.predictor.predict(embedded_data)
        else:
            future = self.predictor.predict(embedded_data)

        return future

    def data_provider(self, dataset, batch_size=None,
                      test=False, d=None, u=None, past=False, info=False):
        """Data provider for model training"""
        N = self.params["timesteps"][0]
        Mth = np.minimum(10, self.params["timesteps"][1])
        if batch_size is None:
            batch_size = self.params["batch_size"]

        with h5py.File(dataset, "r") as f:
            feat = f["descriptor"]
            user = f["user_id"]
            day = f["day"]
            frame = f["frame_id"]
            if info:
                try:
                    gt = f['boundary']
                except Exception as e:
                    print e
                    from extraction_utils import find_gt_from_file
                    gt = find_gt_from_file(frame, user, day)

            def sequence_provider(idx):
                # randomize whether to predict future (0) or past (1)
                s = np.random.choice([0, 1])
                if (test and past) or (
                        not test and ((idx + Mth + N >= len(feat) - 1) or (
                            day[idx] != day[idx + Mth + N]) or (
                                    s and day[idx] == day[idx - (Mth + N)]))):
                    # yield features backwards if future is not available,
                    # and make sure that past is.
                    M = min(idx-N, self.params["timesteps"][1])
                    if day[idx] != day[idx - (M + N)]:
                        M = next(i for i, v in enumerate(np.array(
                                            day[idx - (M + N): idx + 1])[::-1])
                                 if v != day[idx]) - N
                    i = (feat[max(0, idx - N + 1): idx + 1][::-1])
                    t = (feat[max(0, idx - (M + N) + 1):idx - N + 1][::-1])
                    id_seq = frame[idx - N]
                    if info:
                        id_seq = [id_seq,
                                  user[idx - N], day[idx - N], gt[idx - N]]
                else:
                    # yield features forward, to predict the future
                    M = np.minimum(self.params["timesteps"][1],
                                   len(feat) - idx - N - 1)
                    if day[idx] != day[idx + M + N]:
                        M = next(i for i, v in enumerate(day[idx:idx + M + N + 1])
                                 if v != day[idx]) - N
                    i = (feat[idx:idx + N])
                    t = (feat[idx+N:idx + M+N])
                    id_seq = frame[idx + N]    
                    if info:
                        id_seq = [id_seq,
                                  user[idx + N], day[idx + N], gt[idx + N]]

                if len(t) < self.params["timesteps"][1]:
                    #pad shorter sequences for length consistency
                    t = np.pad(t, ((0, self.params["timesteps"][1] - len(t)),
                                   (0, 0)), 'reflect')

                return np.vstack((i, t)), t, id_seq

            if test:
                if d is None and u is None:
                    list_idx = range(len(feat))
                else:
                    list_idx = np.where((day[:] == d)*(user[:] == u))[0]
                list_idx = list_idx[:-(Mth + N)] if not past else list_idx[(N + Mth):]
            else:
                list_idx = range(len(feat))

            keep_going = True
            while keep_going:
                inputs = []
                targets = []
                ids = []

                if test:
                    keep_going = False
                else:
                    list_idx = np.random.permutation(list_idx)

                for ix in (list_idx):
                    if (((ix + Mth + N >= len(feat) - 1) or
                         (day[ix] != day[ix + Mth + N])) and
                            (day[ix] != day[ix - (Mth + N)] or ix < Mth + N)):
                        continue
                    
                    i, t, id_seq = sequence_provider(ix)
                    targets.append(t)
                    inputs.append(i)
                    ids.append(id_seq)

                    if len(inputs) >= batch_size:
                        if test:
                            yield ((np.array(inputs), ids), np.array(targets))
                        else:
                            yield (np.array(inputs), np.array(targets))
                        inputs = []
                        targets = []
                        ids = []

                if len(inputs):
                    if test:
                        yield ((np.array(inputs), ids), np.array(targets))
                    else:
                        yield (np.array(inputs), np.array(targets))

    def train_model(self, epochs=10, steps=200, val_steps=10, initial_epoch=0):
        train_data = self.params["train_dataset"]
        val_data = self.params["val_dataset"]
        callbacks = [TensorBoard(log_dir=self.params['log_dir'])]
        if self.weight_save_callback is not None:
            callbacks.append(self.weight_save_callback)
        if self.other_callbacks is not None:
            [callbacks.append(c) for c in self.other_callbacks]
        self.seq_to_seq.fit_generator(self.data_provider(train_data),
                                      epochs=epochs,
                                      steps_per_epoch=steps,
                                      validation_data=self.data_provider(
                                              val_data),
                                      validation_steps=val_steps,
                                      initial_epoch=initial_epoch,
                                      callbacks=callbacks,
                                      max_queue_size=5)

    def save_model(self, filepath, only_arch=False, only_w=False):
        if only_arch:
            arch = {'seq_to_seq': self.seq_to_seq.to_json(),
                    'encoder': self.encoder.to_json(),
                    'predictor': self.predictor.to_json()}
            np.savez(filepath, arch=arch)
        elif only_w:
            self.seq_to_seq.save_weights(filepath)
        else:
            self.seq_to_seq.save(filepath + 'seq2seq')
            self.encoder.save(filepath + 'encoder')
            self.predictor.save(filepath + 'predictor')

    def load_weights(self, filepath):
        self.seq_to_seq.load_weights(filepath, by_name=True)
        self.encoder.load_weights(filepath, by_name=True)
        self.predictor.load_weights(filepath, by_name=True)


class prunning_SVM(object):
    """Class for SVM supervised pruning"""
    def __init__(self,
                 path_svm='', kernel='poly'):
        self.kernel = kernel
        self.path_svm = path_svm + kernel + 'SVM.pkl'
        self.clf = None

    def load(self):
        # load it again
        with open(self.path_svm, 'rb') as fid:
            print self.path_svm
            self.clf = cPickle.load(fid)
        return self.clf

    def get_datapoints(self, shots, desc):
        from .extraction_utils import get_svm_data
        return get_svm_data(shots, desc)

    def predict(self, data):
        if self.clf is None:
            self.load()
        return self.clf.predict(self.get_datapoints(data[0], data[1]))


class EarlyStoppingTH(Callback):
    """Stop training when a monitored quantity doesn't improve a threshold.
    # Arguments
        monitor: quantity to be monitored.
        th: minimum/maximum quantity to qualify as ok.
        patience: number of epochs with no threshold reaching
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """
    def __init__(self, monitor='val_loss',
                 th=.1, patience=0, verbose=0, mode='auto'):
        import warnings
        super(EarlyStoppingTH, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.threshold = th
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
            else:
                self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current, self.threshold):
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))