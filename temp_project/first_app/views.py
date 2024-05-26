from django.shortcuts import render

from django.http import HttpResponse


def index(request):
    my_dict = {'insert_me': "Hi , i am from views.py file and coming from first_app/index.html"}
    return render(request, 'first_app/index.html', context=my_dict)


def home(request):
    return HttpResponse('welcome to home')
