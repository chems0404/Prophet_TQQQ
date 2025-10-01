# dashboard/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .utils import (
    run_prophet_and_plot,          # TQQQ
    run_upro_prophet_and_plot,     # UPRO
    run_soxl_prophet_and_plot,     # SOXL
    run_qqq_prophet_and_plot,      # QQQ
    run_btc_prophet_and_plot,      # BTC
    run_rhhby_prophet_and_plot,    # RHHBY
    run_prophet_and_plot_tslg,
    run_prophet_and_plot_udow,
    get_tqqq_signal,
    get_upro_signal,
    get_soxl_signal,
    get_qqq_signal,
    get_btc_signal,
    get_rhhby_signal,
    get_tslg_signal,
    get_udow_signal,
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

@login_required
def dashboard_btc_view(request):
    btc_output = run_btc_prophet_and_plot()

    context = {
        'btc_plot_full': btc_output['plot_full'],
        'btc_plot_recent': btc_output['plot_recent'],
        'btc_metrics': btc_output['metrics'],
    }

    return render(request, 'dashboard/dashboard_btc.html', context)
@login_required
def semaforo_btc_view(request):
    context = get_btc_signal()
    return render(request, 'dashboard/semaforo_btc.html', context)

@login_required
def dashboard_tslg_view(request):
    ts_output = run_prophet_and_plot_tslg()
    context = {
        'tslg_plot_full': ts_output['plot_full'],
        'tslg_plot_recent': ts_output['plot_recent'],
        'tslg_metrics': ts_output['metrics'],
    }
    return render(request, 'dashboard/dashboard_tslg.html', context)

@login_required
def semaforo_tslg_view(request):
    context = get_tslg_signal()
    return render(request, 'dashboard/semaforo_tslg.html', context)


@login_required
def dashboard_udow_view(request):
    udow_output = run_prophet_and_plot_udow()
    context = {
        'udow_plot_full': udow_output['plot_full'],
        'udow_plot_recent': udow_output['plot_recent'],
        'udow_metrics': udow_output['metrics'],
    }
    return render(request, 'dashboard/dashboard_udow.html', context)

# >>> ADD: UDOW – Semáforo
@login_required
def semaforo_udow_view(request):
    context = get_udow_signal()
    return render(request, 'dashboard/semaforo_udow.html', context)