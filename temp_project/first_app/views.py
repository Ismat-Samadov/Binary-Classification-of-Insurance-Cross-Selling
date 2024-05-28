from django.shortcuts import render
from first_app.models import User
from django.http import HttpResponse


def index(request):
    users_list = User.objects.order_by('birth_date')
    birth_date_dict = {'birth_date': users_list}
    return render(request, 'first_app/index.html', context=birth_date_dict)


def home(request):
    return HttpResponse('welcome to home')
