from __future__ import unicode_literals
from django.db import models


class Song(models.Model):
	song_title = models.CharField(max_length=250)
	file = models.FileField(upload_to='',default = "null")

	class Meta:
		ordering = ['song_title']

	def __str__(self):
		return self.song_title

class Playlist(models.Model):
    name = models.CharField(max_length = 50)
    songs = models.ManyToManyField(Song)

    def __str__(self):
        return self.name



