# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-train", "--train_data_path", type=str, required=True,
	help="path to training face photos")

# ap.add_argument("-test", "--test_data_path", type=str, required=True,
# 	help="path to testing face photos")

ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")

ap.add_argument("-i", "--input_video_file", type=str, default="0",
	help="path to input video")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

ap.add_argument("-s", "--skip_count", type=float, default=1,
	help="no. of frames to skip prrediction")



def prepare_training_data(data_folder_path):
 
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)
	 
	#list to hold all subject faces
	faces = []
	#list to hold labels for all subjects
	labels = []
	label_map = {}
	 
	#let's go through each directory and read images within it
	for dir_name in dirs:
		dir_path = os.path.join(data_folder_path, dir_name)
		
		label = int(dir_name)

		file_list = os.listdir(dir_path)

		for file_name in file_list:
			if file_name.endswith(".txt"):
				label_name = file_name.split(".")[0]
		label_map[str(label)] = label_name

		for file_name in file_list:


		 
			#our subject directories start with letter 's' so
			#ignore any non-relevant directories if any
			if not file_name.endswith(".jpg"):
				continue;
			 
			#------STEP-2--------
			#extract label number of subject from dir_name
			#format of dir name = slabel
			#, so removing letter 's' from dir_name will give us label
			#label = int(file_name.split("_")[0])
			image_path = os.path.join(dir_path,file_name)

			#read image
			face = cv2.imread(image_path)

			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			 
			#display an image window to show the image 
			print(image_path)
			#cv2.imshow("Training on image...", face)
			#cv2.waitKey(100)
			 
			#add face to list of faces
			faces.append(face)
			#add label for this face
			labels.append(label)

			cv2.destroyAllWindows()
			cv2.waitKey(1)
			cv2.destroyAllWindows()
		 
	return faces, labels, label_map

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


#####################################################Prediction Function
#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img, face_recognizer):
	#make a copy of the image as we don't want to change original image
	img = test_img.copy()
	#detect face from the image
	face, rect = detect_face(img)

	#predict the image using our face recognizer 
	label = face_recognizer.predict(face)
	#get name of respective label returned by face recognizer
	label_text = str(label)
	 
	#draw a rectangle around face detected
	draw_rectangle(img, rect)
	#draw name of predicted person
	draw_text(img, label_text, rect[0], rect[1]-5)
	 
	return img

def predict_from_face(face_img, face_recognizer, label_map):

	#predict the image using our face recognizer 
	label = face_recognizer.predict(face)
	print(label)
	#get name of respective label returned by face recognizer
	label_text = label_map[str(label[0])]
	 
	#draw a rectangle around face detected
	#draw_rectangle(img, rect)
	#draw name of predicted person
	#draw_text(img, label_text, rect[0], rect[1]-5)
	 
	return label_text



args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


######################################################## Data Preparation

print("Preparing data...")
faces, labels, label_map = prepare_training_data(args["train_data_path"])
print(label_map)
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

###################################################create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create(neighbors=10, grid_x=8, grid_y=8)

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createEigenFaceRecognizer()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

########################################################Perform Prediction
print("Predicting images...")

if(args["input_video_file"]=="0"):
	vs = cv2.VideoCapture(1)
else:
	vs = cv2.VideoCapture(args["input_video_file"])

time.sleep(2.0)
start_frame_number = 10
vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
	
#vs = VideoStream(src=0).start()


# loop over the frames from the video stream

if vs.isOpened(): # try to get the first frame
	rval, frame = vs.read()
else:
	print("VideoCapture not opened")
	rval = False

skip_frame_interval = args["skip_count"]


start_time = time.time()
frame_count = 0
while rval:
	frame_count+=1

	rval, frame = vs.read()
	if(rval==False): break
	skip_frame_interval -=1
	if(skip_frame_interval>0):
		continue
	
	skip_frame_interval = args["skip_count"]
		
	#print (frame)
	if(frame.any()==None): continue
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	if len(detections) > 0:
		# extract the confidence (i.e., probability) associated with the
		# prediction
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

			text_to_show = predict_from_face(face, face_recognizer, label_map)

			cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
			cv2.putText(frame, text_to_show, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

end_time = time.time()

print("Processing rate: "+str(frame_count/int(end_time-start_time))+"fps")
# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()





# #load test images
# test_img1 = cv2.imread("test-data/test1.jpg")
# test_img2 = cv2.imread("test-data/test2.jpg")

# #perform a prediction
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
# print("Prediction complete")

# #display both images
# cv2.imshow(subjects[1], predicted_img1)
# cv2.imshow(subjects[2], predicted_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



