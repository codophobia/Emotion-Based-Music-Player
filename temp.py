import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
import dlib
import os
import math
import numpy as np 
from PIL import Image

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

def normalize(xpoints,ypoints,xmean,ymean):
	xstd = np.std(xpoints)
	ystd = np.std(ypoints)
	nxpoints = []
	nypoints = []
	for x in xpoints:
		t = float((x-xmean))
		nxpoints.append(t)
	for y in ypoints:
		t = float((y-ymean))
		nypoints.append(t)
	return nxpoints,nypoints

def euclid(x1,y1,x2,y2):
	return float(math.sqrt(((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2))))

def vector(xpoints,ypoints,nxpoints,nypoints,xmean,ymean):
	v = []
	for nx,ny,x,y in zip(nxpoints,nypoints,xpoints,ypoints):
		v.append(nx)
		v.append(ny)
		dis = euclid(x,y,xmean,ymean)
		v.append(dis)
	return v

detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("./music/static/shape_predictor_68_face_landmarks.dat") #Landmark 
d = {"HA":0,"SA":1,"SU":2,"AN":3}
def landmark_detector(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)
	xpoints = []
	ypoints = []
	detections = detector(clahe_image, 1) #Detect the faces in the image

	for k,d in enumerate(detections): #For each detected face
		shape = predictor(clahe_image, d) #Get coordinates
		for i in range(1,68): #There are 68 landmark points on each face
			cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
			xpoints.append(float(shape.part(i).x))
			ypoints.append(float(shape.part(i).y))
	xmean = np.mean(xpoints)
	ymean = np.mean(ypoints)
	nxpoints,nypoints = normalize(xpoints,ypoints,xmean,ymean)
	feature = vector(xpoints,ypoints,nxpoints,nypoints,xmean,ymean)
	return feature
def trainfiles():
	training_data = []
	training_label = []
	for files in os.listdir('./music/static/train'):
		for file in os.listdir('./music/static/train/%s'%files):
			fname = './music/static/train/%s/%s'%(files,file)
			angle = file[6] 
			exp = file[4] + file[5]
			if angle == 'S':
				image = cv2.imread(fname)
				v = landmark_detector(image)
				if len(v) > 0 and (exp == 'HA' or exp == 'SA' or exp == 'SU' or exp == "AN"):
					training_data.append(v)
					training_label.append(d[exp])

	return training_data,training_label
def testfiles():
	test_data = []
	images = []
	for files in os.listdir('./music/static/test'):
		for file in os.listdir('./music/static/test/%s'%files):
			fname = './music/static/test/%s/%s'%(files,file)
			angle = file[6] 
			exp = file[4] + file[5]
			if angle == 'S':
				image = cv2.imread(fname)
				v = landmark_detector(image)
				if len(v) > 0 and exp == 'HA' or exp == 'SA' or exp == "SU" or exp == "AN":
					test_data.append(v)
					images.append(image)

	return test_data,images

clf = SVC(kernel='linear', probability=True, tol=1e-3)
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

	train()

display()