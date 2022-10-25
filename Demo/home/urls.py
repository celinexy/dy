from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect),
    path('detect/',views.detect),
    path('index/', views.index),
    path('test/', views.test),
    path('video_feed/',views.video_feed,name='video_feed'),
]