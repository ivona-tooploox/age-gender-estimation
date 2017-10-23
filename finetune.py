import pandas as pd
import logging
import argparse
import os
import numpy
from sklearn.model_selection import GridSearchCV
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from wide_resnet import WideResNet
from utils import mk_dir, load_data

logging.basicConfig(level=logging.DEBUG)


class Schedule:
    def __init__(self, nb_epochs):
        self.epochs = nb_epochs

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return 0.1
        elif epoch_idx < self.epochs * 0.5:
            return 0.02
        elif epoch_idx < self.epochs * 0.75:
            return 0.004
        return 0.0008


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Freeze layers for training")
    args = parser.parse_args()
    return args

def create_model():
    model = WideResNet(image_size, depth=depth, k=k)()

    # Load weights
    weight_file = os.path.join("pretrained_models", "weights.18-4.06.hdf5")
    model.load_weights(weight_file, by_name=True)

    # # set the first 50 layers 
    # # to non-trainable (weights will not be updated)
    # print(len(model.layers))
    # if freeze_layers > 0 :
    #     for layer in model.layers[:freeze_layers]:
    #         layer.trainable = False

    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=["binary_crossentropy"],
                  metrics=['accuracy'])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    return model

args = get_args()
input_path = args.input
batch_size = args.batch_size
nb_epochs = args.nb_epochs
freeze_layers = args.freeze_layers
depth = args.depth
k = args.width
validation_split = args.validation_split

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load dataset
logging.debug("Loading data...")
image, gender, age, _, image_size, _ = load_data(input_path)
#reshape image data which is in format 64 x 64 x 3 <-> 12288 
X_data = image
n_samples = len(X_data)
# X_data = image.reshape((n_samples, 12288))
print(gender, 'raw y_data_g')
y_data_g = np_utils.to_categorical(gender, 2)
y_data_a = np_utils.to_categorical(age, 101)

print('X shape', X_data.shape)
print('y_g', y_data_g)
print('y_a', y_data_a)


model = KerasClassifier(build_fn = create_model, verbose=0)

# define the grid search parameters
batch_size = [10, 20, 32, 64]
epochs = [10, 15, 20, 25, 30]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result = grid.fit(X_data, gender)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

    

    # logging.debug("Saving model...")
    # mk_dir("models")
    # with open(os.path.join("models", "WRN_{}_{}.json".format(depth, k)), "w") as f:
    #     f.write(model.to_json())

    # mk_dir("checkpoints")
    # callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs)),
    #              ModelCheckpoint("checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
    #                              monitor="val_loss",
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode="auto")
    #              ]

    # logging.debug("Running training...")
    # # print('length of X', len(X_data))
    # # print('length of y_data_g', y_data_g)
    # # print('length of y_data_a', len(y_data_a))
    # hist = model.fit(X_data, [y_data_g, y_data_a], batch_size=batch_size, epochs=nb_epochs, callbacks=callbacks,
    #                  validation_split=validation_split)

    # logging.debug("Saving weights...")
    # model.save_weights(os.path.join("models", "WRN_{}_{}.h5".format(depth, k)), overwrite=True)
    # pd.DataFrame(hist.history).to_hdf(os.path.join("models", "history_{}_{}.h5".format(depth, k)), "history")


