from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from accounts.views import CustomLoginView, signup
from dashboard.views import (
    elegir_dashboard_view,
    dashboard_view,
    dashboard_upro_view,
    recalcular_view,
    explicacion_view,
)

urlpatterns = [
    path('admin/', admin.site.urls),

    # Login personalizado
    path('accounts/login/', CustomLoginView.as_view(), name='login'),
    path('accounts/signup/', signup, name='signup'),
    path('accounts/', include('django.contrib.auth.urls')),

    # Redirige raíz a login
    path('', RedirectView.as_view(pattern_name='login', permanent=False)),

    # Página de selección de dashboard tras login
    path('dashboard/', elegir_dashboard_view, name='elegir_dashboard'),

    # Dashboards separados
    path('dashboard/tqqq/', dashboard_view, name='dashboard_tqqq'),
    path('dashboard/upro/', dashboard_upro_view, name='dashboard_upro'),

    # Recalcular y explicación
    path('recalcular/', recalcular_view, name='recalcular'),
    path('como-funciona/', explicacion_view, name='explicacion'),
]
