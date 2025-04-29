# proyecto3/urls.py
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from accounts.views import CustomLoginView, signup
from dashboard.views import dashboard_view


urlpatterns = [
    path('admin/', admin.site.urls),

    # Login personalizado
    path('accounts/login/', CustomLoginView.as_view(), name='login'),
    # Registro de nuevos usuarios
    path('accounts/signup/', signup, name='signup'),
    # El resto de URLs de auth: logout, password_reset, password_change…
    path('accounts/', include('django.contrib.auth.urls')),

    # Redirigir la raíz a login (opcional)
    path('', RedirectView.as_view(pattern_name='login', permanent=False)),
    path('dashboard/', dashboard_view, name='dashboard')
]
