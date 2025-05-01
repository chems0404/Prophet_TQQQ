# dashboard/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .utils import run_prophet_and_plot, run_upro_prophet_and_plot
from django.contrib.auth.decorators import login_required

@login_required
def dashboard_view(request):
    """
    Vista protegida para mostrar el dashboard de Prophet.
    Ejecuta el pipeline y pasa las métricas y gráficos al template.
    """
    tqqq_output = run_prophet_and_plot()
    upro_output = run_upro_prophet_and_plot()

    context = {
        'tqqq_metrics': tqqq_output['metrics'],
        'tqqq_plot_full': tqqq_output['plot_full'],
        'tqqq_plot_recent': tqqq_output['plot_recent'],
        
        'upro_metrics': upro_output['metrics'],
        'upro_plot_full': upro_output['plot_full'],
        'upro_plot_recent': upro_output['plot_recent'],
    }
    return render(request, 'dashboard/dashboard.html', context)

@login_required
def recalcular_view(request):
    """
    Fuerza recálculo del modelo borrando el caché.
    Redirige al dashboard con datos actualizados.
    """
    run_prophet_and_plot.cache_clear()
    run_upro_prophet_and_plot.cache_clear()
    return redirect('dashboard')

@login_required
def explicacion_view(request):
    return render(request, 'dashboard/explicacion.html')

@login_required
def elegir_dashboard_view(request):
    return render(request, 'dashboard/elegir_dashboard.html')

@login_required
def dashboard_upro_view(request):
    upro_output = run_upro_prophet_and_plot()

    context = {
        'upro_plot_full': upro_output['plot_full'],
        'upro_plot_recent': upro_output['plot_recent'],
        'upro_metrics': upro_output['metrics'],
    }

    return render(request, 'dashboard/dashboard_upro.html', context)


