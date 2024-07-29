from pathlib import Path
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from data_generator import prepare_train_test, create_dataset
from models.cnn3head_lstm import CNN3Head_LSTM

data_dir = Path(r'../data/eeg').resolve()
train, val, test = prepare_train_test(data_dir=data_dir)
print(len(train))

file_path = str(Path(r'../saved_models/cnn3head_lstm.h5'))
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
# callbacks_list = [checkpoint, early, redonplat]  # early

# decide where to save the tensorboard logs
log_dir = Path(r'logs\cnn3head_lstm')
log_path = log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# instantiate callbacks
tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
callbacks = [tensorboard_callback,
             checkpoint,
             early,
             redonplat
             ]

# split train, valid and test data
X, y = create_dataset(train)
val_X, val_y = create_dataset(val)
test_X, test_y = create_dataset(test)


# run the cnn basic model with only 1 head
'''
model = CNN1Head(model_name='CNN1Head_all',
                 epochs=2,
                 learning_rate=0.001,
                 batch_size=16,
                 metric='sparse_categorical_accuracy'
                 )

model.build_model()

# hist = model.fit_model(train_dl)
hist = model.fit(X, y, val_X, val_y)  # validation_data=(val_X,val_y))
'''

# run the improved cnn model with 3 head
'''
model = CNN3Head(model_name='CNN3Head_all',
                 epochs=2,
                 learning_rate=0.001,
                 batch_size=16,
                 metric='sparse_categorical_accuracy'
                 )
model.build_model()

# hist = model.fit_model(train_dl)
hist = model.fit_model(X, y, val_X, val_y)  # validation_data=(val_X,val_y))
'''

# run the optimized 3-head cnn model with an additional lstm layer
model = CNN3Head_LSTM(model_name='CNN3Head_lstm_all',
                 epochs=20,
                 learning_rate=0.001,
                 batch_size=16,
                 metric='sparse_categorical_accuracy'
                 )

# build the model
model.build_model()

# history information
hist = model.fit_model(X, y, val_X, val_y)  # validation_data=(val_X,val_y))

# evaluate the model
acc = model.evaluate(test_X, test_y)
print("acc = ", acc)
print("done")
print(hist.history)
