from third_app import views
from django.urls import path
from django.contrib import admin
from django.urls import path
from django.conf.urls import include

urlpatterns = [path('data/', views.data, name='data'),
               path('', views.form_submit_view, name='form_submit_view'),
               ]
