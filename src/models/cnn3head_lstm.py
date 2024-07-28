from tensorflow.keras import optimizers
from code.models.base_model import BaseModel
from pathlib import Path

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model

import sys
sys.path.append("../")
from code.data_generator import create_dataset, prepare_train_test

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

saved_model_dir = Path(r'.\saved_models')


class CNN3Head_LSTM(BaseModel):
    def build_model(self):


        #input_layer = Input(batch_shape=(self.batch_size, 3000, 1))
        input_layer = Input(shape=(3000, 1))

        # Head 1
        conv1 = Conv1D(filters=64,
                       kernel_size=3,
                       strides=1,
                       activation='relu',
                       padding='same')(input_layer)
        drop1 = Dropout(0.5)(conv1)
        maxpool1 = MaxPooling1D(pool_size=2,
                                #strides=4,
                                padding='same')(drop1)

        # Head 2
        conv2 = Conv1D(filters=64,
                       kernel_size=5,
                       strides=1,
                       activation='relu',
                       padding='same')(input_layer)
        drop2 = Dropout(0.5)(conv2)
        maxpool2 = MaxPooling1D(pool_size=2,
                                #strides=4,
                                padding='same')(drop2)

        # Head 3
        conv3 = Conv1D(filters=64,
                       kernel_size=11,
                       strides=1,
                       activation='relu',
                       padding='same')(input_layer)
        drop3 = Dropout(0.5)(conv3)
        maxpool3 = MaxPooling1D(pool_size=2,
                                #strides=4,
                                padding='same')(drop3)

        # Concatenate
        merge1 = Concatenate()([maxpool1, maxpool2, maxpool3])

        # additional cnn after concatenation
        conv11 = Conv1D(filters=128,
                        kernel_size=4,
                        strides=1,
                        activation='relu')(merge1)
        maxpool11 = MaxPooling1D(pool_size=4,
                                 strides=2)(conv11)

        # LSTM
        lstm1 = LSTM(128)(merge1)

        dense1 = Dense(100, activation='relu')(lstm1)
        outputs = Dense(5, activation='softmax')(dense1)

        model = Model(inputs=input_layer, outputs=outputs)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.Adam(self.learning_rate),
                      metrics=['sparse_categorical_accuracy']
                      )
        self.model = model
        plot_model(model,
                   to_file='./model_diagrams/model_cnn3head_lstm.png',
                   show_shapes=False,
                   show_layer_names=False
                   )
        return model

    def fit_model(self, train_X, train_y, val_X, val_y):
        return self.model.fit(train_X, train_y,
                              # validation_split=0.4,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              verbose=True,
                              callbacks=self.callbacks
                              )

    def evaluate(self, testX, testy):
        _, accuracy = self.model.evaluate(testX, testy, batch_size=self.batch_size, verbose=2)
        return accuracy


if __name__ == '__main__':
    data_dir = Path(r'../../data/demo_eeg').resolve()
    train, val, test = prepare_train_test(data_dir=data_dir)
    print(len(train))

    X, y = create_dataset(train[0:3])
    val_X, val_y = create_dataset(val[0:1])
    # train_dl = DataGenerator(train[0:3])
    # evaluate_model(X, y, val_X, val_y)
    model = CNN3Head_LSTM(model_name='CNN3Head_train10_test0',
                          epochs=1,
                          learning_rate=0.001,
                          batch_size=16,
                          metric='sparse_categorical_accuracy'
                          )
    model.build_model()
    hist = model.fit_model(X, y, val_X, val_y)  # validation_data=(val_X,val_y))

    # hist = model.fit(X,y, val_X, val_y)
    print(hist.history)
    print('done')
