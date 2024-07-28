
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
import itertools

def convert_to_onehot(array):
    n_values = np.max(array) + 1
    return np.eye(n_values)[array]


class BaseModel:
    def __init__(self,
                 config={},
                 model_name = 'base_model',
                 saved_model_dir = Path(r'out/saved_models'),
                 figure_dir = Path(r'out/figures'),
                 tensorboard_log_dir=Path(r'logs/fit'),
                 n_timesteps=3000,
                 n_channels=1,
                 n_outputs=5,
                 epochs=20,
                 batch_size=32,
                 learning_rate=0.005,
                 metric='sparse_categorical_accuracy'):
        self.config = config
        self.model_name = model_name
        self.saved_model_dir = saved_model_dir
        self.figure_dir = figure_dir
        self.tensorboard_log_dir = tensorboard_log_dir

        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.metric = metric
        self.callbacks = self.create_callbacks()

        # n_timesteps, n_features, n_outputs

    def create_callbacks(self):
        file_path = str(self.saved_model_dir/f'{self.model_name}.h5')
        checkpoint = ModelCheckpoint(file_path, monitor=self.metric, verbose=1, save_best_only=True, mode='max')  # mode='max'
        early_stopping = EarlyStopping(monitor=self.metric, mode="max", patience=20, verbose=2)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor=self.metric, mode="max", patience=20, verbose=2)
        # callbacks_list = [checkpoint, early, reduce_lr_on_plateau]  # early
        log_dir = self.tensorboard_log_dir
        log_path = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
        callbacks = [tensorboard_callback,
                     checkpoint,
                     early_stopping,
                     reduce_lr_on_plateau]
        return callbacks

    def build_model(self):
        self.model = Sequential()
        return NotImplementedError('Any model inherit the base model should implement this part')

    def plot_roc_auc(self, y_preds, y_tests, classes_to_plot=[0,1,2,3,4]):
        model_name = self.model_name
        y_preds_oh = convert_to_onehot(y_preds)
        y_tests_oh = convert_to_onehot(y_tests)
        print(y_preds_oh)
        colors = ['red', 'aqua', 'darkorange', 'cornflowerblue', 'green', 'deeppink']
        labels = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3','Sleep stage R']
        plt.figure()
        for c in classes_to_plot:
            fpr, tpr, _ = roc_curve(y_tests_oh[:, c], y_preds_oh[:, c])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC of class {labels[c]} = {roc_auc:0.2f})", color=colors[c])
        plt.legend(loc="lower right")
        #     plt.show()
        # plt.savefig(str(self.figure_dir / f'{model_name}_auc.png'))
        plt.savefig(f'{model_name}_auc.png')

    def plot_confusion_matrix(self, cm, class_names, model_name):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(i, j, "{:.2f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.figure_dir / f'{model_name}_cm.png')
        # plt.savefig("./plots/{}_cm.png".format(model_name))


if __name__=='__main__':
    model = BaseModel()
    y_pred = np.random.randint(0, 6, 100)
    y_test = np.random.randint(0, 6, 100)
    model.plot_roc_auc(y_preds=y_pred, y_tests=y_test)
