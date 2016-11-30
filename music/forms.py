from django import forms
from music.models import Song

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Song
        fields = ('song_title', 'file', )