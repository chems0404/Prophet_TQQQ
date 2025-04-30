from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User


class CustomAuthForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False, initial=False)


class SignUpForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(widget=forms.PasswordInput, label='Repite la contraseña')
    invitation_code = forms.CharField(
        label='Código de invitación',
        max_length=20,
        help_text='Requiere autorización previa'
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2', 'invitation_code']

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError("Las contraseñas no coinciden")
        return cd['password2']

    def clean_invitation_code(self):
        code = self.cleaned_data.get("invitation_code")
        if code != "TQQQ2025":  # <-- Cambiá este código cuando quieras
            raise forms.ValidationError("Código de invitación incorrecto.")
        return code
