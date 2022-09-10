# import packages
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np

import argparse
import cv2
import os

# parse arguments
ap = argparse.ArgumentParser() 
ap.add_argument("-i","--img", required=True, type=str, default="default-img-path", help=" provide path to input image")
args = vars(ap.parse_args())

print("loading face detector model...")
net = cv2.dnn.readNet(os.path.sep.join(["facemodel", "deploy.prototxt"]), os.path.sep.join(["facemodel", "resnet50_model.caffemodel"]))
model = load_model("facemodel"])

image = cv2.imread(args[img])
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 117, 123))
net.setInput(blob)
detections = net.forward()

print("computing face detections...")

for i in range(0, detections.shape[2]):
	# get confidence fromm detection
	confi = detections[0, 0, i, 2]
	if confi > 0.5:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# determine face ROI bounding box and make sure they are not out of bounds
		(x0, y0, x1, y2) = box.astype("int")

		x0, y0 = max(0, x0), max(0,y0)
		x1, y2 = min(w-1, x1), min(h-1,y2)

		face = cv2.resize(face, (244,244)) # resize to 244, 244
		face = preprocess_input(img_to_array(face))
		face = np.expand_dims(face, axis=0)
		(mask, withoutMask) = model.predict(face)[0]

		# show probability in the label
		label = "{}: {:.2f}%".format("Mask" if mask > withoutMask else "No Mask", max(mask, withoutMask) * 100)
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		cv2.putText(image, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.rectangle(image, (x0, y0), (x1, y2), color, 2)

# show output image
cv2.imshow("Output", image)
cv2.waitKey(0)
