from __future__ import unicode_literals
from django.db import models

class Playlist(models.Model):
	genre = models.CharField(max_length = 200)

	def __str__(self):
		return self.genre

class Song(models.Model):
	song_title = models.CharField(max_length=250)
	file = models.FileField(upload_to='/',default = "null")

	def __str__(self):
		return self.song_title
