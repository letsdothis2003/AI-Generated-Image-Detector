#Fahim Tanvir and Ahmed Ali
#CSCI-367 Term Project
#Please check the directory portion below 

import struct
from pathlib import Path
import numpy as np


def load_images(path):
    path = Path(path)

    with path.open("rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:  #that identifies an image file according to the IDX format
            raise ValueError(f"Wrong magic number for images: {magic}")

        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        #read the raw image bytes after the header
        data = np.frombuffer(f.read(), dtype=np.uint8)

    images = data.reshape(num_images, rows, cols)
    return images


def load_labels(path):

    path = Path(path)

    with path.open("rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:  # that identifies a label file in the IDX format
            raise ValueError(f"Wrong magic number for labels: {magic}")

        num_items = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)

    #quick falier check
    if data.shape[0] != num_items:
        raise ValueError("Label count does not match header information")

    return data

 #You can change directory of the data in this portion of the code. Data direct is implement to work with workflow. 
def load_mnist_dataset(data_dir="data"):
    data_dir = Path(data_dir)

    train_images_path = data_dir / "train-images-idx3-ubyte"
    train_labels_path = data_dir / "train-labels-idx1-ubyte"
    test_images_path = data_dir / "t10k-images-idx3-ubyte"
    test_labels_path = data_dir / "t10k-labels-idx1-ubyte"


    X_train = load_images(train_images_path).astype(np.float32) / 255.0
    y_train = load_labels(train_labels_path).astype(np.int64)

    X_test = load_images(test_images_path).astype(np.float32) / 255.0
    y_test = load_labels(test_labels_path).astype(np.int64)

    return X_train, y_train, X_test, y_test
