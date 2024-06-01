from django.shortcuts import render
from django.http import HttpResponseRedirect
from third_app import forms
from third_app.models import User

def data(request):
    users_list = User.objects.order_by('first_name')
    first_name_dict = {'first_name': users_list}
    return render(request, 'third_app/index.html', context=first_name_dict)
def form_submit_view(request):
    form = forms.Form_Submit()
    if request.method == 'POST':
        form = forms.Form_Submit(request.POST)
        if form.is_valid():
            print("Data Validation")
            print("First Name: " + form.cleaned_data['first_name'])
            print("Last Name: " + form.cleaned_data['last_name'])
            print("Email: " + form.cleaned_data['email'])
            print("Birth Date: " + str(form.cleaned_data['birth_date']))
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            email = form.cleaned_data['email']
            birth_date = form.cleaned_data['birth_date']
            p = User(first_name=first_name, last_name=last_name, email=email, birth_date=birth_date)
            p.save()
            return HttpResponseRedirect('/register/data/')
    return render(request, 'third_app/form_page.html', {'form': form})
