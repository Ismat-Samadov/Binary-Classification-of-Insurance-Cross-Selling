from django.shortcuts import render

from django.http import HttpResponse


def help(request):
    my_dict = {'help_page': "Hi , i am from views.py file and coming from second_app/index.html"}
    return render(request, 'second_app/index.html', context=my_dict)

