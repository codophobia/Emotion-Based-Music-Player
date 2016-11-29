from django.conf.urls import *
from music import views
urlpatterns = [
    url(r'^$',views.home,name = 'home'),
   	url(r'^playlist/$',views.playlist,name='playlist'),
   	url(r'^songs/$',views.songs,name='songs'),
   	url(r'^next/$',views.next,name='next'),
   	url(r'^prev/$',views.prev,name='prev'),
   	url(r'^psongs/$',views.psongs,name='psongs'),
   	url(r'^createplaylist/$',views.createplaylist,name='createplaylist'),
   	url(r'^deletesong/$',views.deletesong,name='deletesong'),
   	url(r'^addplaylistsongs/$',views.addplaylistsongs,name='addplaylistsongs'),
   	url(r'^addtoplaylist/$',views.addtoplaylist,name='addtoplaylist'),
   	url(r'^deleteplaylistsong/$',views.deleteplaylistsong,name='deleteplaylistsong'),
   	url(r'^deleteplaylist/$',views.deleteplaylist,name='deleteplaylist'),
      url(r'^mooddetect/$',views.mooddetect,name='mooddetect'),

]