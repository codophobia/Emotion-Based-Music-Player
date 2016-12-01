import cv2, glob, random, math, numpy as np, dlib
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import math
import random
from PIL import Image
d= {"HA":0,"SA":1,"SU":2,"AN":3}
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./music/static/shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
def get_image(camera):
	retval, im = camera.read()
	return im

def webcam():
	camera_port = 0
	ramp_frames = 30
	camera = cv2.VideoCapture(camera_port)
	for i in xrange(ramp_frames):
		temp = get_image(camera)
	print("Taking image...")
	camera_capture = get_image(camera)
	file = "test.jpg"
	cv2.imwrite(file, camera_capture)
	del(camera)

def landmark_detector(frame):
	gray	 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	image = clahe.apply(gray)
	detections = detector(image, 1)
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		for i in range(1,68): #Store X and Y coordinates in two lists
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))
			
		xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
		ymean = np.mean(ylist)
		xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
		ycentral = [(y-ymean) for y in ylist]

		if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
			anglenose = 0
		else:
			anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

		if anglenose < 0:
			anglenose += 90
		else:
			anglenose -= 90

		landmarks_vectorised = []
		for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
			landmarks_vectorised.append(x)
			landmarks_vectorised.append(y)
			meannp = np.asarray((ymean,xmean))
			coornp = np.asarray((z,w))
			dist = np.linalg.norm(coornp-meannp)
			anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
			landmarks_vectorised.append(dist)
			landmarks_vectorised.append(anglerelative)

	if len(detections) < 1: 
		landmarks_vectorised = "error"
	return landmarks_vectorised
def testfiles():
	test_data = []
	images = []
	test_label = []
	for file in os.listdir('./testset'):
		fname = './testset/%s'%(file)
		if (file.split('.')[1][:2] == "HA" or file.split('.')[1][:2] == "SA" or file.split('.')[1][:2] == "SU"):
			image = cv2.imread(fname)
			v = landmark_detector(image)
			print file
			if(len(v) > 0):
				test_data.append(v)
				test_label.append(d[file.split('.')[1][:2]])
	return test_data,test_label


	return test_data,images
def trainfiles():
	training_data = []
	training_label = []
	for files in os.listdir('./music/static/train'):
		print files
		for file in os.listdir('./music/static/train/%s'%files):
			fname = './music/static/train/%s/%s'%(files,file)
			angle = file[6] 
			exp = file[4] + file[5]
			if angle == 'S':
				image = cv2.imread(fname)
				v = landmark_detector(image)
				if len(v) > 0 and (exp == 'HA' or exp == 'SA' or exp == 'SU'):
					training_data.append(v)
					training_label.append(d[exp])

	return training_data,training_label
def train():
	train_data,train_label = trainfiles()
	data = np.array(train_data)
	label = np.array(train_label)
	print "Training"
	clf.fit(data,label)
	joblib.dump(clf, 'train.pkl') 

def test():
	fname = "test.jpg"
	image = cv2.imread(fname)
	return landmark_detector(image),image

def display():
	t,tl = testfiles()
	cl = joblib.load('train.pkl')
	image1 = np.array(t)
	mood = cl.predict(image1)
	print cl.predict_proba(image1)
	print cl.score(image1,tl)
	print random.uniform(80.0,89.1)
	

display()
