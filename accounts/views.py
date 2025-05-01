from django.contrib.auth.views import LoginView
from .forms import CustomAuthForm, SignUpForm
from django.shortcuts import render, redirect
from django.contrib.auth import login

# Código secreto de invitación (puedes moverlo a settings.py si deseas más adelante)
INVITATION_CODE = 'TQQQ2025'


class CustomLoginView(LoginView):
    authentication_form = CustomAuthForm
    template_name = 'registration/login.html'

    def form_valid(self, form):
        remember = form.cleaned_data.get('remember_me')
        if remember:
            self.request.session.set_expiry(1209600)  # 2 semanas
        else:
            self.request.session.set_expiry(0)  # hasta cerrar navegador
        login(self.request, form.get_user())  # <- Esto asegura redirección manual
        return redirect('elegir_dashboard')  # Redirige a la pantalla de elección


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            if form.cleaned_data['invitation_code'] != INVITATION_CODE:
                form.add_error('invitation_code', 'El código de invitación es incorrecto.')
            else:
                user = form.save(commit=False)
                user.set_password(form.cleaned_data['password'])
                user.save()
                login(request, user)
                return redirect('elegir_dashboard')  # Redirige a la pantalla de elección
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})
