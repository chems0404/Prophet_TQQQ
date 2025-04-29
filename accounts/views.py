# accounts/views.py
from django.contrib.auth.views import LoginView
from .forms import CustomAuthForm

class CustomLoginView(LoginView):
    authentication_form = CustomAuthForm
    template_name = 'registration/login.html'

    def form_valid(self, form):
        # Si marcó “remember_me”, dura 2 semanas (1209600 s), 
        # si no, expira al cerrar navegador
        remember = form.cleaned_data.get('remember_me')
        if remember:
            self.request.session.set_expiry(1209600)
        else:
            self.request.session.set_expiry(0)
        return super().form_valid(form)


from django.shortcuts import render, redirect
from .forms import SignUpForm

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})
