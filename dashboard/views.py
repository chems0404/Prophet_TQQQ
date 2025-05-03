# dashboard/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .utils import (
    run_prophet_and_plot,
    run_upro_prophet_and_plot,
    run_soxl_prophet_and_plot,
    run_qqq_prophet_and_plot,
    run_rhhby_prophet_and_plot,  # <-- Añadido
    get_tqqq_signal,
    get_upro_signal,
    get_soxl_signal,
    get_qqq_signal,
    get_rhhby_signal            # <-- Añadido
)

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
    run_prophet_and_plot.cache_clear()
    run_upro_prophet_and_plot.cache_clear()
    return redirect('dashboard_tqqq')  # Aquí corregido


@login_required
def dashboard_soxl_view(request):
    soxl_output = run_soxl_prophet_and_plot()

    context = {
        'soxl_plot_full': soxl_output['plot_full'],
        'soxl_plot_recent': soxl_output['plot_recent'],
        'soxl_metrics': soxl_output['metrics'],
    }

    return render(request, 'dashboard/dashboard_soxl.html', context)


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

@login_required
def semaforo_tqqq_view(request):
    context = get_tqqq_signal()
    return render(request, 'dashboard/semaforo_tqqq.html', context)

@login_required
def semaforo_upro_view(request):
    context = get_upro_signal()
    return render(request, 'dashboard/semaforo_upro.html', context)

@login_required
def semaforo_soxl_view(request):
    context = get_soxl_signal()
    return render(request, 'dashboard/semaforo_soxl.html', context)

@login_required
def dashboard_qqq_view(request):
    qqq_output = run_qqq_prophet_and_plot()

    context = {
        'qqq_plot_full': qqq_output['plot_full'],
        'qqq_plot_recent': qqq_output['plot_recent'],
        'qqq_metrics': qqq_output['metrics'],
    }

    return render(request, 'dashboard/dashboard_qqq.html', context)
@login_required
def semaforo_qqq_view(request):
    context = get_qqq_signal()
    return render(request, 'dashboard/semaforo_qqq.html', context)


@login_required
def dashboard_rhhby_view(request):
    rhhby_output = run_rhhby_prophet_and_plot()

    context = {
        'rhhby_plot_full': rhhby_output['plot_full'],
        'rhhby_plot_recent': rhhby_output['plot_recent'],
        'rhhby_metrics': rhhby_output['metrics'],
    }

    return render(request, 'dashboard/dashboard_rhhby.html', context)

@login_required
def semaforo_rhhby_view(request):
    context = get_rhhby_signal()
    return render(request, 'dashboard/semaforo_rhhby.html', context)

