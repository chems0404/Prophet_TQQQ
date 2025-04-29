# accounts/forms.py
from django import forms
from django.contrib.auth.forms import AuthenticationForm

class CustomAuthForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False, initial=False)

from django import forms
from django.contrib.auth.models import User

class SignUpForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(widget=forms.PasswordInput, label='Repite la contraseña')

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2']

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError("Las contraseñas no coinciden")
        return cd['password2']
