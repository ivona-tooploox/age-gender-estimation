import pandas as pd
import logging
import argparse
import os
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
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
    args = parser.parse_args()
    return args



    



def main():
    args = get_args()
    input_path = args.input
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    depth = args.depth
    k = args.width
    validation_split = args.validation_split

    logging.debug("Loading data...")

    image, gender, age, _, image_size, _ = load_data(input_path)
    X_data = image
    # y_data_g = np_utils.to_categorical(gender, 2)
    # y_data_a = np_utils.to_categorical(age, 101)

    #Load weights
    weight_file = os.path.join("pretrained_models", "weights.18-4.06.hdf5")
    
    # Build model
    model = WideResNet(image_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    print('Model layers number', len(model.layers))
    for layer in model.layers:
        print(layer)

    #save output as numpy array
    bottleneck_features = model.predict(X_data)

    np.save(open('bottleneck_features.npy', 'wb'), bottleneck_features[1])

    

    

if __name__ == '__main__':
    main()
