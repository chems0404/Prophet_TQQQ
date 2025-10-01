from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from accounts.views import CustomLoginView, signup
from dashboard.views import (
    elegir_dashboard_view,
    dashboard_view,
    dashboard_upro_view,
    dashboard_soxl_view,
    dashboard_qqq_view,
    dashboard_rhhby_view,
    dashboard_btc_view,            # ← NUEVO
    dashboard_tslg_view,
    dashboard_udow_view,
    explicacion_view,
    semaforo_tqqq_view,
    semaforo_upro_view,
    semaforo_soxl_view,
    semaforo_qqq_view,
    semaforo_rhhby_view,
    semaforo_btc_view,
    semaforo_tslg_view,
    semaforo_udow_view,
)

urlpatterns = [
    path('admin/', admin.site.urls),

    # Login y registro personalizado
    path('accounts/login/', CustomLoginView.as_view(), name='login'),
    path('accounts/signup/', signup, name='signup'),
    path('accounts/', include('django.contrib.auth.urls')),

    # Redirección raíz al login
    path('', RedirectView.as_view(pattern_name='login', permanent=False)),

    # Página de selección de dashboard
    path('dashboard/', elegir_dashboard_view, name='elegir_dashboard'),

    # Dashboards
    path('dashboard/tqqq/', dashboard_view, name='dashboard_tqqq'),
    path('dashboard/upro/', dashboard_upro_view, name='dashboard_upro'),
    path('dashboard/soxl/', dashboard_soxl_view, name='dashboard_soxl'),
    path('dashboard/qqq/', dashboard_qqq_view, name='dashboard_qqq'),
    path('dashboard/rhhby/', dashboard_rhhby_view, name='dashboard_rhhby'),
    path('dashboard/btc/', dashboard_btc_view, name='dashboard_btc'),
    path('dashboard/tslg/', dashboard_tslg_view, name='dashboard_tslg'), # ← NUEVO
    path('dashboard/udow/', dashboard_udow_view, name='dashboard_udow'),  # ← NUEVO

    # Señales tipo semáforo
    path('semaforo/tqqq/', semaforo_tqqq_view, name='semaforo_tqqq'),
    path('semaforo/upro/', semaforo_upro_view, name='semaforo_upro'),
    path('semaforo/soxl/', semaforo_soxl_view, name='semaforo_soxl'),
    path('semaforo/qqq/', semaforo_qqq_view, name='semaforo_qqq'),
    path('semaforo/rhhby/', semaforo_rhhby_view, name='semaforo_rhhby'),
    path('semaforo/btc/', semaforo_btc_view, name='semaforo_btc'),
    path('semaforo/tslg/', semaforo_tslg_view, name='semaforo_tslg'), 
    path('semaforo/udow/', semaforo_udow_view, name='semaforo_udow'),    # ← NUEVO

    # Otros
    path('como-funciona/', explicacion_view, name='explicacion'),
]
