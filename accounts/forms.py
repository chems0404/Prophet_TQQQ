from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from .models import InvitationCode, Profile

class CustomAuthForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False, initial=False)

class SignUpForm(forms.ModelForm):
    email = forms.EmailField(required=True)
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(widget=forms.PasswordInput, label='Repite la contraseña')
    invitation_code = forms.CharField(
        label='Código de invitación',
        max_length=32,
        help_text='Requiere autorización previa'
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2', 'invitation_code']

    def clean_password2(self):
        cd = self.cleaned_data
        if cd.get('password') != cd.get('password2'):
            raise forms.ValidationError("Las contraseñas no coinciden.")
        return cd['password2']

    def clean_invitation_code(self):
        code_str = self.cleaned_data.get("invitation_code")
        try:
            inv = InvitationCode.objects.get(code=code_str, is_active=True)
        except InvitationCode.DoesNotExist:
            raise forms.ValidationError("Código de invitación incorrecto o está desactivado.")
        return inv  # devolvemos la instancia, no el string

    def save(self, commit=True):
        # Creamos el usuario, seteamos email y la contraseña hasheada
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.set_password(self.cleaned_data['password'])

        if commit:
            user.save()
            # Asociamos el perfil con la empresa del código
            inv: InvitationCode = self.cleaned_data['invitation_code']
            Profile.objects.create(user=user, company=inv.company)
            # Incrementamos contador de usos
            inv.used_count += 1
            inv.save(update_fields=['used_count'])

        return user
