# dashboard/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .utils import run_prophet_and_plot

@login_required
def dashboard_view(request):
    """
    Vista protegida para mostrar el dashboard de Prophet.
    Ejecuta el pipeline y pasa las métricas y gráficos al template.
    """
    data = run_prophet_and_plot()
    return render(request, 'dashboard/dashboard.html', data)

@login_required
def recalcular_view(request):
    """
    Fuerza recálculo del modelo borrando el caché.
    Redirige al dashboard con datos actualizados.
    """
    run_prophet_and_plot.cache_clear()  # limpia el cache
    return redirect('dashboard')  # vuelve a cargar el dashboard
