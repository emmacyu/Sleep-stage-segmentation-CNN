from tensorflow.keras import optimizers
from pathlib import Path
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import sys
from tensorflow.keras.models import Sequential
sys.path.append("../")
from code.data_generator import create_dataset, prepare_train_test
from code.models.base_model import BaseModel


import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

saved_model_dir = Path(r'.\saved_models')


class CNN1Head(BaseModel):
    def build_model(self):
        n_timesteps = self.n_timesteps
        n_features = self.n_channels
        n_outputs = self.n_outputs
        model = Sequential(layers=[
            Input(shape=(n_timesteps, n_features)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            Dropout(0.5),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(n_outputs, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(self.learning_rate),
                      metrics=['sparse_categorical_accuracy'])
        self.model = model
        plot_model(model,
                   to_file='model_cnn1head.png',
                   show_shapes=False,
                   show_layer_names=False
                   )
        return model

    def fit(self, trainX, trainy, valX, valy):
        return self.model.fit(trainX, trainy,
                       validation_split=0.4,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=2,
                       callbacks=self.callbacks)

    def evaluate(self, testX, testy):
        _, accuracy = self.model.evaluate(testX, testy, batch_size=self.batch_size, verbose=2)
        return accuracy

    def plot_confusion_matrix(self, y_true, y_pred):
        pass


if __name__ == '__main__':
    data_dir = Path(r'../../data/demo_eeg').resolve()
    train, val, test = prepare_train_test(data_dir=data_dir)
    print(len(train))

    X, y = create_dataset(train[0:3])
    val_X, val_y = create_dataset(val[0:1])
    # train_dl = DataGenerator(train[0:3])
    # evaluate_model(X, y, val_X, val_y)
    model = CNN1Head(model_name='CNN1Head',
                     epochs=2,
                     learning_rate=0.001,
                     batch_size=16,
                     metric='sparse_categorical_accuracy'
                     )
    model.build_model()
    hist = model.fit(X, y, val_X, val_y)  # validation_data=(val_X,val_y))

    # hist = model.fit(X,y, val_X, val_y)
    print(hist.history)
    print('done')
