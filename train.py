import pickle
import tflearn
from tflearn.data_utils import shuffle
from nn_model import generate_model
import os
import numpy as np

s_epoch = 5
i_epoch = 5
batch = 1000

print("Loading Datasets ...")
train_dataset_file = "extra_dataset.p"
validation_dataset_file = "test_dataset.p"
train_ds = pickle.load(open(train_dataset_file, "rb"))
validation_ds = pickle.load(open(validation_dataset_file, "rb"))
model_name = "lemodel.tfl"

print("Processing Datasets ...")
X = np.array(train_ds["X"], dtype='float16') / 256
Y = np.array(train_ds["Y"], dtype='float64')
X_test = np.array(validation_ds["X"], dtype='float16') / 256
Y_test = np.array(validation_ds["Y"], dtype='float64')
(train_ds, validation_ds) = (0, 0)
X, Y = shuffle(X, Y)

print("Generating Convolutional Model ...")
model = generate_model()
if model_name in os.listdir():
    print("Loading Weights ...")
    model.load(model_name)

print("Training ...")
for i in range(0, s_epoch):
    model.fit(X, Y, n_epoch=i_epoch, shuffle=True, validation_set=(X_test, Y_test),
    show_metric=True, batch_size=batch, run_id='digits_cnn')

    print("Saving Weights ...")
    model.save(model_name)
print("Done")
