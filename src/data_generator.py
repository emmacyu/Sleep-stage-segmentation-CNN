# import the necessary packages
import functools
from pathlib import Path
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split


def rescale_array(X):
    '''
    normalize the input
    '''
    X = X / 20
    X = np.clip(X, -5, 5)
    return X


class DataGenerator(Sequence):
    def __init__(self, file_list, batch_size=32):
        '''
        initialization
        '''
        self.file_list = file_list
        self.idx_file_mapping, self.num_samples = self._create_idx_file_mapping()
        self.batch_size = batch_size

    def _create_idx_file_mapping(self):
        '''
        create index and file mapping
        '''
        count = 0
        mapping = {}
        for file_path in self.file_list:
            with np.load(file_path) as d:
                size = d['y'].shape[0]
                mapping[(count, count + size)] = file_path
            count += size
        return mapping, count

    def _find_file_path(self, idx):
        '''
        locate the file
        '''
        for idx_range, file_path in self.idx_file_mapping.items():
            start, end = idx_range
            if start <= idx and idx <= end:
                in_file_idx = idx - start
                return (in_file_idx, file_path)

    def __len__(self):
        '''
        get number of batches
        '''
        return int(np.ceil(self.num_samples / self.batch_size))

    @functools.lru_cache(maxsize=128)
    def _read_file(self, file_path):
        ''' 
        load the file
        '''
        with np.load(file_path) as d:
            return d['x'], d['y']

    def __getitem__(self, idx):
        '''
        getters
        '''
        in_file_idx, file_path = self._find_file_path(idx)
        X, y = self._read_file(file_path)
        # y = to_categorical(y,5)
        X_batch = X[in_file_idx:(in_file_idx + self.batch_size)]
        X_batch = rescale_array(X_batch)
        y_batch = y[in_file_idx:(in_file_idx + self.batch_size)]
        # y_batch = to_categorical(y_batch, 5)
        return X_batch, y_batch  # ,[None]


class SeqDataGenerator(Sequence):
    def __init__(self, file_list, batch_size=4, time_steps=10):
        '''
        initialization of necessary parameters
        '''
        self.file_list = file_list
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.idx_file_mapping, self.num_batches = self._create_idx_file_mapping()

        print(f'time steps:{time_steps}')
        print(f'num batches:{self.num_batches}')

    def _create_idx_file_mapping(self):
        '''
        create index and file mapping
        '''
        count = 0
        mapping = {}
        for file_path in self.file_list:
            with np.load(file_path) as d:
                size = d['y'].shape[0]
                if size == 0:
                    print(f'bad file: {file_path}')
                    continue
                n_batch = size // (self.batch_size * self.time_steps)
                # mapping[(count, count + size)] = file_path
                mapping[(count, count + n_batch)] = file_path
            # count += size
            count += n_batch
        return mapping, count

    def _find_file_path(self, idx):
        '''
        locate the file
        '''
        for idx_range, file_path in self.idx_file_mapping.items():
            start, end = idx_range
            if start <= idx and idx < end:
                in_file_idx = idx - start
                return (in_file_idx, file_path)

    def __len__(self):
        '''
        get number of batches
        '''
        return self.num_batches
        # return int(np.floor(self.num_samples / (self.batch_size*self.time_steps))

    @functools.lru_cache(maxsize=128)
    def _read_file(self, file_path):
        ''' 
        load the file
        '''
        with np.load(file_path) as d:
            return d['x'], d['y']

    def __getitem__(self, idx):
        '''
        getters
        '''
        in_file_idx, file_path = self._find_file_path(idx)
        X, y = self._read_file(file_path)
        ind0 = in_file_idx * (self.batch_size * self.time_steps)
        ind1 = (in_file_idx + 1) * self.batch_size * self.time_steps
        seq_X = X[ind0:ind1]
        print(f'ind0:{ind0}, ind1:{ind1},X shape:{seq_X.shape}')
        seq_X = seq_X.reshape(self.batch_size, self.time_steps, 3000)
        seq_X = np.expand_dims(seq_X, -1)
        seq_X = rescale_array(seq_X)
        seq_y = y[ind0:ind1]
        seq_y = seq_y.reshape(self.batch_size, self.time_steps)
        seq_y = np.expand_dims(seq_y, -1)
        print(f'seq_X shape: {seq_X.shape}')
        print(f'seq_y shape: {seq_y.shape}')
        if seq_y.shape[2] == 0:
            print(seq_y)
        return seq_X, seq_y
        # seq_y = y[in_file_idx:(in_file_idx + self.batch_size)]
        # seq_y = tf.keras.utils.to_categorical(seq_y, num_classes=5)
        # return np.expand_dims(seq_X,0) , np.expand_dims(seq_y, 0)


class SeqOneOutGenerateor(Sequence):
    def __init__(self, file_list, batch_size=4, time_steps=10):
        '''
        initialization of necessary parameters
        '''
        self.file_list = file_list
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.idx_file_mapping, self.num_batches = self._create_idx_file_mapping()

        print(f'time steps:{time_steps}')
        print(f'num batches:{self.num_batches}')

    def _create_idx_file_mapping(self):
        '''
        create index and file mapping
        '''
        count
        count = 0
        mapping = {}
        for file_path in self.file_list:
            with np.load(file_path) as d:
                size = d['y'].shape[0]
                if size == 0:
                    print(f'bad file: {file_path}')
                    continue
                n_batch = size // (self.batch_size * self.time_steps)
                # mapping[(count, count + size)] = file_path
                mapping[(count, count + n_batch)] = file_path
            # count += size
            count += n_batch
        return mapping, count

    def _find_file_path(self, idx):
        '''
        locate the file
        '''
        for idx_range, file_path in self.idx_file_mapping.items():
            start, end = idx_range
            if start <= idx and idx < end:
                in_file_idx = idx - start
                return (in_file_idx, file_path)

    def __len__(self):
        '''
        get number of batches
        '''
        return self.num_batches
        # return int(np.floor(self.num_samples / (self.batch_size*self.time_steps))

    @functools.lru_cache(maxsize=128)
    def _read_file(self, file_path):
        ''' 
        load the file
        '''
        with np.load(file_path) as d:
            return d['x'], d['y']

    def __getitem__(self, idx):
        '''
        getters
        '''
        in_file_idx, file_path = self._find_file_path(idx)
        X, y = self._read_file(file_path)
        ind0 = in_file_idx * (self.batch_size * self.time_steps)
        ind1 = (in_file_idx + 1) * self.batch_size * self.time_steps
        seq_X = X[ind0:ind1]
        print(f'ind0:{ind0}, ind1:{ind1},X shape:{seq_X.shape}')
        seq_X = seq_X.reshape(self.batch_size, self.time_steps, 3000)
        seq_X = np.expand_dims(seq_X, -1)
        seq_X = rescale_array(seq_X)
        # seq_y = y[ind0:ind1]
        seq_y = np.zeros((self.batch_size, 1))
        for i_batch in range(self.batch_size):
            seq_y[i_batch] = y[ind0 + i_batch * self.time_steps]
        # seq_y = y[ind1]
        # seq_y = seq_y.reshape(self.batch_size, self.time_steps)
        # seq_y = np.expand_dims(seq_y, -1)
        print(f'seq_X shape: {seq_X.shape}')
        print(f'seq_y shape: {seq_y.shape}')
        # if seq_y.shape[2] == 0:
        #     print(seq_y)
        return seq_X, seq_y


def create_dataset(data_files):
    '''
    split data into small sample frames
    As the sample size is 30s and frequency is 100Hz, the the length for each sample frame will be 3000
    '''
    Xs = np.empty([0, 3000, 1])
    ys = np.empty([0, ])
    for dfile in data_files:
        with np.load(dfile) as d:
            x = d['x']
            y = d['y']
            #rint(f'x shape:{x.shape}')
            #print(f'y shape:{y.shape}')
            Xs = np.vstack((Xs, x))
            ys = np.hstack((ys, y))
    return Xs, ys


def prepare_train_test(data_dir=Path(r'..\data\eeg_fpz_cz')):
    '''
    split the train, valid and test datasets
    '''
    print('start')
    data_files = list(data_dir.glob('*.npz'))
    print(f'number of data files: {len(data_files)}')

    train_val, test = train_test_split(data_files, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)
    return train, val, test


if __name__ == "__main__":
    '''
    this main class is used for testing purpuses
    '''
    print('start')
    train, val, test = prepare_train_test()
    print(len(train))
