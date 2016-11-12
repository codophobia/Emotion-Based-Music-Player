from django.conf.urls import *
from music import views
urlpatterns = [
    url(r'^$',views.home,name = 'home'),
   
]