from tensorflow.keras import optimizers
from models.base_model import BaseModel
from pathlib import Path

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import sys
sys.path.append("../")
from code.data_generator import create_dataset, prepare_train_test
from keras.utils.vis_utils import plot_model

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

saved_model_dir = Path(r'.\saved_models')


class CNN3Head(BaseModel):
    def build_model(self):
        n_timesteps = self.n_timesteps
        n_features = self.n_channels
        n_outputs = self.n_outputs

        # head 1
        inputs1 = Input(shape=(n_timesteps, n_features))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        # head 2
        inputs2 = Input(shape=(n_timesteps, n_features))
        conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)

        # head 3
        inputs3 = Input(shape=(n_timesteps, n_features))
        conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])

        # interpretation
        dense1 = Dense(100, activation='relu')(merged)
        outputs = Dense(n_outputs, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # save a plot of the model
        # plot_model(model, show_shapes=True, to_file='multichannel.png')
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.Adam(self.learning_rate),
                      metrics=['sparse_categorical_accuracy']
                      )
        self.model = model

        plot_model(model,
                   to_file='./model_diagrams/model_cnn3head.png',
                   show_shapes=False,
                   show_layer_names=False
                   )

        return model

    def fit_model(self, train_X, train_y, val_X, val_y):
        return self.model.fit([train_X, train_X, train_X], train_y,
                              # validation_split=0.4,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              verbose=2,
                              callbacks=self.callbacks)

    def evaluate(self, testX, testy):
        _, accuracy = self.model.evaluate([testX, testX, testX], testy, batch_size=self.batch_size, verbose=2)
        return accuracy


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 2, 25, 16
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], 5
    # head 1
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # head 2
    inputs2 = Input(shape=(n_timesteps, n_features))
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # head 3
    inputs3 = Input(shape=(n_timesteps, n_features))
    conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # save a plot of the model
    # plot_model(model, show_shapes=True, to_file='multichannel.png')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit([trainX, trainX, trainX], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.save(saved_model_dir / 'cnn3heads_origin.h5')
    # evaluate model
    _, accuracy = model.evaluate([testX, testX, testX], testy, batch_size=batch_size, verbose=0)

    return accuracy


if __name__ == '__main__':
    data_dir = Path(r'../../data/demo_eeg').resolve()
    train, val, test = prepare_train_test(data_dir=data_dir)
    print(len(train))

    X, y = create_dataset(train[0:3])
    val_X, val_y = create_dataset(val[0:1])
    # train_dl = DataGenerator(train[0:3])
    # evaluate_model(X, y, val_X, val_y)
    model = CNN3Head(model_name='CNN3Head_train10_test0',
                     epochs=2,
                     learning_rate=0.001,
                     batch_size=16,
                     metric='sparse_categorical_accuracy'
                     )
    model.build_model()
    hist = model.fit_model(X, y, val_X, val_y)  # validation_data=(val_X,val_y))

    # hist = model.fit(X,y, val_X, val_y)
    print(hist.history)
    print('done')
