# proyecto3/urls.py
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from accounts.views import CustomLoginView, signup
from dashboard.views import dashboard_view, recalcular_view, explicacion_view




urlpatterns = [
    path('admin/', admin.site.urls),

    # Login personalizado
    path('accounts/login/', CustomLoginView.as_view(), name='login'),
    path('accounts/signup/', signup, name='signup'),
    path('accounts/', include('django.contrib.auth.urls')),

    path('', RedirectView.as_view(pattern_name='login', permanent=False)),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('recalcular/', recalcular_view, name='recalcular'),
    path('como-funciona/', explicacion_view, name='explicacion'),
]

