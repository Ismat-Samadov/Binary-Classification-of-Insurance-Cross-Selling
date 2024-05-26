from first_app import views
from django.urls import path
from django.contrib import admin
from django.urls import path
from django.conf.urls import include

urlpatterns = [path('', views.index, name='index'),
               path('', views.home, name='home'),
               ]
