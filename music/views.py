from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from models import Song
def home(request):
	song = Song.objects.all()
	context = {"songs":song} 
	return render(request,'music/base.html',context)

