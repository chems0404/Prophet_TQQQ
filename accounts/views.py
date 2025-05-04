from django.contrib.auth.views import LoginView
from django.shortcuts import render, redirect
from django.contrib.auth import login
from .forms import CustomAuthForm, SignUpForm

class CustomLoginView(LoginView):
    authentication_form = CustomAuthForm
    template_name = 'registration/login.html'

    def form_valid(self, form):
        remember = form.cleaned_data.get('remember_me')
        if remember:
            self.request.session.set_expiry(1209600)
        else:
            self.request.session.set_expiry(0)
        login(self.request, form.get_user())
        return redirect('elegir_dashboard')


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            # Aquí ejecutamos todo el save() que creamos en el formulario:
            #   1) Hashea y guarda la contraseña
            #   2) Asigna email y username
            #   3) Crea el Profile con la empresa correcta
            #   4) Incrementa used_count del InvitationCode
            user = form.save()
            login(request, user)
            return redirect('elegir_dashboard')
    else:
        form = SignUpForm()

    return render(request, 'registration/signup.html', {'form': form})
