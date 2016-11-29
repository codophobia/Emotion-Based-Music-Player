from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from models import Song,Playlist
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core import serializers
import json
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
import dlib
import os
import math
import numpy as np 
from PIL import Image

hpy = []
sd = []

def home(request):
	global hcount,scount
	hcount = 0
	scount = 0
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
		if(isplay == 0):
			next_song = Song.objects.all().filter(song_title__gt = currsong)[:1]
			if not next_song:
				next_song = Song.objects.all()[:1]
		else:
			pname = request.POST['pname']
			if(len(happy) > 0 or len(sad) > 0):
				print json.loads(happy)
				print json.loads(sad)
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
			next_song = Song.objects.all().filter(song_title__lt = currsong)[:1]
		else:
			pname = request.POST['pname']
			p = Playlist.objects.get(name=pname)
			next_song = p.songs.all().filter(song_title__lt = currsong)[:1]
		data = serializers.serialize('json', next_song)
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
d = {"AF":0,"AN":1,"DI":2,"HA":3,"NE":4,"SA":5,"SU":6}
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

def test(cnt):
	fname = "./music/static/detect/test.jpg" 
	image = cv2.imread(fname)
	return landmark_detector(image)
hcount = 0
scount = 0
@csrf_exempt
def mooddetect(request):
	global hcount
	global scount
	if request.method == 'POST':
		cnt = request.POST['cnt']
		webcam(cnt)
		image = test(cnt)
		try:
			cl = joblib.load('train.pkl') 
			image = np.array(image)
			mood = cl.predict(image)
			m = mood[0]
			if(m == 0):
				hcount = hcount + 1
			else:
				scount = scount + 1
			if(int(cnt) == 0):
				if (hcount >= scount):
					x = "happy"
					p = Playlist.objects.get(name='happy')
					songs = p.songs.all().order_by('?')
				else:
					x = "sad"
					p = Playlist.objects.get(name='sad')
					songs = p.songs.all().order_by('?')
				if(m == 0):
					hcount = 1
					scount = 0
				else:
					scount = 1
					hcount = 0	
				print x
				data = json.loads(serializers.serialize('json', songs))
				d = {}
				d['result'] = data
				d['mood'] = m
				d['hcount'] = hcount
				d['scount'] = scount
				d['fmood'] = x
				data = json.dumps(d)
			else:
				d = {}
				d['mood'] = m
				d['fmood'] = "happy"
				d['hcount'] = hcount
				d['scount'] = scount
				data = json.dumps(d)
			return HttpResponse(data, content_type="application/json")
		except:
			d = {}
			d['mood'] = "not found"
			data = json.dumps(d)
			return HttpResponse(data, content_type="application/json")