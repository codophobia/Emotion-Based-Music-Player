from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.conf import settings
from models import Song,Playlist
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core import serializers
import json
import cv2
from music.forms import DocumentForm
from django.core.files.storage import FileSystemStorage
from sklearn.svm import SVC
from sklearn.externals import joblib
import dlib
import os
import math
import numpy as np 
from PIL import Image

hpy = []
sd = []
rmouth = 0
ratio = 0
hcount = 0
scount = 0
sucount = 0


@csrf_exempt
def clearcount(request):
	print "hi"
	global hcount,scount,sucount
	hcount = scount = sucount = 0
@csrf_exempt
def upload(request):
	if request.method == 'POST' and request.FILES['picture']:
		myfile = request.FILES['picture']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)
		f = uploaded_file_url.split('/')
		fname = f[2]
		s = Song()
		s.song_title = filename
		s.file = fname
		s.save()
		return redirect('home')
def home(request):
	global hcount,scount
	hcount = 0
	scount = 0
	sucount = 0
	song = Song.objects.all()
	context = {"songs":song} 
	return render(request,'music/base.html',context)

@csrf_exempt
def playlist(request):
	playlists = Playlist.objects.all()
	data = serializers.serialize('json', playlists)
	return HttpResponse(data, content_type="application/json")
	
@csrf_exempt
def songs(request):
	song = Song.objects.all()
	data = serializers.serialize('json', song)
	return HttpResponse(data, content_type="application/json")

@csrf_exempt
def next(request):
	if request.method == 'POST':
		isplay = int(request.POST['isplay'])
		currsong = request.POST['currsong']
		happy = request.POST.getlist('happy[]')
		sad = request.POST.getlist('sad[]')
		print isplay
		if(isplay == 0):
			next_song = Song.objects.all().filter(song_title__gt = currsong)[:1]
			if not next_song:
				next_song = Song.objects.all()[:1]
		else:
			pname = request.POST['pname']
			p = Playlist.objects.get(name=pname)
			next_song = p.songs.all().filter(song_title__gt = currsong)[:1]
			if not next_song:
				p = Playlist.objects.get(name=pname)
				next_song = p.songs.all()[:1]
		data = serializers.serialize('json', next_song)
		return HttpResponse(data, content_type="application/json")

@csrf_exempt
def prev(request):
	if request.method == 'POST':
		isplay = int(request.POST['isplay'])
		currsong = request.POST['currsong']
		if(isplay == 0):
			song = Song.objects.all().filter(song_title__lt = currsong).last()
			next_song = Song.objects.get(song_title = song)
		else:
			pname = request.POST['pname']
			p = Playlist.objects.get(name=pname)
			next_song = p.songs.all().filter(song_title__lt = currsong).last()
		data = serializers.serialize('json', next_song)
		print data
		return HttpResponse(data, content_type="application/json")
@csrf_exempt
def psongs(request):
	if request.method == 'POST':
		name = (request.POST['name'])
		p = Playlist.objects.get(name=name)
		songs = p.songs.all()
		data = serializers.serialize('json', songs)
		return HttpResponse(data, content_type="application/json")

@csrf_exempt
def createplaylist(request):
	if request.method == 'POST':
		name = (request.POST['name'])
		p = Playlist()
		p.name = name
		p.save()
		return HttpResponse("success")

@csrf_exempt
def deletesong(request):
	if request.method == 'POST':
		song = (request.POST['song'])
		Song.objects.filter(song_title = song).delete()
		return HttpResponse("success")

@csrf_exempt
def deleteplaylistsong(request):
	if request.method == 'POST':
		song = (request.POST['song'])
		playlist = request.POST['playlist']
		s = Song.objects.get(song_title=song)
		p = Playlist.objects.get(name=playlist)
		p.songs.remove(s)
		return HttpResponse("success")

@csrf_exempt
def addplaylistsongs(request):
	if request.method == 'POST':
		name = (request.POST['name'])
		songs = Song.objects.all()
		data = serializers.serialize('json', songs)
		return HttpResponse(data, content_type="application/json")

@csrf_exempt
def deleteplaylist(request):
	if request.method == 'POST':
		song = (request.POST['playlist'])
		Playlist.objects.filter(name=song).delete()
		return HttpResponse("success")
@csrf_exempt
def addtoplaylist(request):
	if request.method == 'POST':
		name = request.POST['name']
		songs = request.POST.getlist('songs[]')
		p = Playlist.objects.get(name=name)
		for s in songs:
			s = Song.objects.get(song_title = s)
			p.songs.add(s)
		return HttpResponse("success")


def get_image(camera):
	retval, im = camera.read()
	return im

def webcam(cnt):
	camera_port = 0
	ramp_frames = 30
	camera = cv2.VideoCapture(camera_port)
	for i in xrange(ramp_frames):
		temp = get_image(camera)
	print("Taking image...")
	camera_capture = get_image(camera)
	file = "./music/static/detect/test.jpg"
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
clf = SVC(kernel='linear', probability=True, tol=1e-3)
d = {"AF":0,"AN":1,"DI":2,"HA":3,"NE":4,"SA":5,"SU":6}
def landmark_detector(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)
	xpoints = []
	ypoints = []
	global ratio
	detections = detector(clahe_image, 1) #Detect the faces in the image

	for k,d in enumerate(detections): #For each detected face
		shape = predictor(clahe_image, d) #Get coordinates
		x1 = euclid(shape.part(48).x,shape.part(48).y,shape.part(54).x,shape.part(54).y)
		y1 = 0.5*euclid(shape.part(43).x,shape.part(43).y,shape.part(47).x,shape.part(47).y)
		y2 = 0.5*euclid(shape.part(38).x,shape.part(38).y,shape.part(40).x,shape.part(40).y)
		y3 = euclid(shape.part(27).x,shape.part(27).y,shape.part(33).x,shape.part(33).y)
		y4 = euclid(shape.part(21).x,shape.part(21).y,shape.part(22).x,shape.part(22).y)
		y5 = euclid(shape.part(39).x,shape.part(39).y,shape.part(42).x,shape.part(42).y)
		y6 = euclid(shape.part(50).x,shape.part(50).y,shape.part(57).x,shape.part(57).y)
		reye = (y1+y2)/y3
		rmouth = x1/(y1+y2)
		reyebrow = (y4/y5)
		ratio = max(x1,y6)/min(x1,y6)
		print reye
		print rmouth
		print reyebrow
		for i in range(1,68): #There are 68 landmark points on each face
			cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
			xpoints.append(float(shape.part(i).x))
			ypoints.append(float(shape.part(i).y))
	xmean = np.mean(xpoints)
	ymean = np.mean(ypoints)
	nxpoints,nypoints = normalize(xpoints,ypoints,xmean,ymean)
	feature = vector(xpoints,ypoints,nxpoints,nypoints,xmean,ymean)
	return feature

def test(cnt):
	fname = "./music/static/detect/test.jpg" 
	image = cv2.imread(fname)
	return landmark_detector(image)

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
				if len(v) > 0 and (exp == 'HA' or exp == 'SA' or exp == 'SU'):
					training_data.append(v)
					training_label.append(d[exp])

	return training_data,training_label
def train():
	train_data,train_label = trainfiles()
	data = np.array(train_data)
	label = np.array(train_label)
	clf.fit(data,label)
	joblib.dump(clf, 'train.pkl') 
@csrf_exempt
def testfiles(request):
	test_data = []
	images = []
	test_label = []
	total = 0
	d = {"HA":0,"SA":1,"SU":2}
	ishappy = issad = issur = 0
	nhappy = nsad = nsur = 0
	for files in os.listdir('./music/static/test'):
		for file in os.listdir('./music/static/test/%s'%files):
			fname = './music/static/test/%s/%s'%(files,file)
			angle = file[6] 
			exp = file[4] + file[5]
			if angle == 'S':
				image = cv2.imread(fname)
				v = landmark_detector(image)
				if len(v) > 0 and exp == 'HA' or exp == 'SU' or exp == "SA":
					if(exp == "HA"):
						total = total + 1
					test_data.append(v)
					images.append(image)
					test_label.append(d[exp])
	for (data,label,image) in zip(test_data,test_label,images):
		clf = joblib.load('train.pkl')
		image1 = np.array(data)
		mood = clf.predict(image1)
		ans = mood[0]
		s = ""
		v = landmark_detector(image)
		if(label == 2):
			if(ans == 2):
				issur = issur+1
			else:
				nsur = nsur + 1
		elif label == 0:
			if(ans == 0):
				ishappy = ishappy+1
			else:
				nhappy = nhappy+1
		elif label == 1:
			if(ans == 1):
				issad = issad+1
			else:
				nsad = nsad+1
	print issur,ishappy,issad,nsur,nhappy,nsad
	ha = float(float(ishappy)/(nhappy+ishappy))*100
	sa = float(float(issad)/(nsad+issad))*100
	sua = float(float(issur)/(nsur+issur))*100
	td = np.array(test_data)
	score = clf.score(td,test_label)*100
	d = {}
	d['ha'] = ha
	d['sa'] = sa
	d['sua'] = sua
	d['score'] = score
	data = json.dumps(d)
	return HttpResponse(data, content_type="application/json")


@csrf_exempt
def mooddetect(request):
	global hcount,ratio
	global scount,sucount
	if request.method == 'POST':
		cnt = request.POST['cnt']
		webcam(cnt)
		image = test(cnt)
		try:
			cl = joblib.load('train.pkl') 
			image = np.array(image)
			mood = cl.predict(image)
			m = mood[0]
			flag = 0
			if(ratio < 2):
				sucount = sucount + 1
				flag = 1
			elif(m == 0):
				hcount = hcount + 1
			else:
				scount = scount + 1
			if(int(cnt) == 0):
				print hcount,scount,sucount
				mx = max(hcount,scount,sucount)
				if (mx == hcount):
					x = "happy"
					p = Playlist.objects.get(name='happy')
					songs = p.songs.all().order_by('?')
				elif(mx == scount):
					x = "sad"
					p = Playlist.objects.get(name='sad')
					songs = p.songs.all().order_by('?')
				else:
					x = "surprise"
					p = Playlist.objects.get(name='surprise')
					songs = p.songs.all().order_by('?')
				if(flag == 1):
					hcount = scount = 0
					sucount = 1
				elif(m == 0):
					hcount = 1
					scount = sucount = 0
				else:
					scount = 1
					hcount = sucount = 0	
				data = json.loads(serializers.serialize('json', songs))
				d = {}
				d['result'] = data
				d['mood'] = m
				d['flag'] = flag
				d['hcount'] = hcount
				d['scount'] = scount
				d['fmood'] = x
				d['sucount'] = sucount
				data = json.dumps(d)
			else:
				d = {}
				d['mood'] = m
				d['flag'] = flag
				d['fmood'] = "happy"
				d['hcount'] = hcount
				d['scount'] = scount
				d['sucount'] = sucount
				data = json.dumps(d)
			return HttpResponse(data, content_type="application/json")
		except:
			d = {}
			d['mood'] = "not found"
			data = json.dumps(d)
			return HttpResponse(data, content_type="application/json")