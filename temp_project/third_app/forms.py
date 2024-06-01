from django import forms


class Form_Submit(forms.Form):
    first_name = forms.CharField()
    last_name = forms.CharField()
    email = forms.EmailField()
    birth_date = forms.DateField()
