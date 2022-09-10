# import packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50 #Deep convolutional neural network
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt # plot our training curves
from imutils import paths # find list images in dataset
import numpy as np # linear algebra
import argparse # arguments
import os

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, type=str,  default="default-dataset-path", help="provide path to input dataset")
args = vars(ap.parse_args())

# initialize hyperparameters: initial learning rate, # of epochs to train for, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


print("[loading images...")
imgPaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
for imgPath in imgPaths:
	labels.append(imgPath.split(os.path.sep)[-2]) #extract class label from filename
	data.append(preprocess_input(img_to_array(load_img(imgPath, target_size=(224, 224))))) #preprocess the image

data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels, partition data into 20% training and 80% testing
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42) #partition data

tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=Input(shape=(224, 224, 3)),
    pooling=None,
)

# load the ResNet50, construct head model
baseModel = resnet50(weights="imagenet", 
    input_shape=(224, 224, 3),
    pooling=avg,
	include_top = False,
	input_tensor = Input(shape=(224, 224, 3))
    classes=2)
headModel = AveragePooling2D(pool_size=(7, 7))(baseModel.output) #pool data
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over and free all layers in the base model so that they will not be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compiling model through adam - stochastic gradient descent
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) 
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Data augmentation and train the network head
aug = ImageDataGenerator(rotation_range=40, zoom_range=0.25, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
H = model.fit(x=aug.flow(trainX, trainY, BS), validation_data=(testX, testY), validation_steps=len(testX) // BS,
	steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=BS)
probIdxs = np.argmax(predIdxs, axis=1) #idices of the maximum predicted probabilities


# saving serial model to disk
print("saving mask detector model...")
model.save("../MaskDetector.model", "h5")# saving serial model to disk

# printing classification report
print(classification_report(testY.argmax(axis=1), probIdxs, lb.classes_))

# plot training loss and accuracy data
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("../plot.png")
