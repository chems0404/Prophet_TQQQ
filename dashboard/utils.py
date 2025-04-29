import pandas as pd
import numpy as np
import matplotlib
# Forzar backend no-GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from prophet import Prophet
from ta.momentum import StochasticOscillator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from pandas.tseries.offsets import BDay
from functools import lru_cache
from pathlib import Path
from django.conf import settings
BASE_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1) 
def run_prophet_and_plot():
    # 1. Cargar y preparar TQQQ
    tqqq = (
        pd.read_csv(BASE_DIR / 'data' / 'TQQQ_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'TQQQ.Close': 'y'})
    )
    stoch = StochasticOscillator(
        close=tqqq['y'], high=tqqq['TQQQ.High'], low=tqqq['TQQQ.Low'],
        window=14, smooth_window=3
    )
    tqqq['stoch_k'] = stoch.stoch()
    tqqq['stoch_d'] = stoch.stoch_signal()
    scaler = StandardScaler()
    tqqq['volume_scaled'] = scaler.fit_transform(tqqq[['TQQQ.Volume']])

    # 2. Cargar y preparar QQQ
    qqq = (
    pd.read_csv(BASE_DIR / 'data' / 'QQQ_data.csv', parse_dates=['Date'])
    .rename(columns={'Date': 'ds', 'QQQ.Close': 'qqq_close'})
)
    qqq['qqq_return'] = np.log(qqq['qqq_close'] / qqq['qqq_close'].shift(1))

    # 3. Merge + rezagos
    df = pd.merge(tqqq, qqq[['ds','qqq_close','qqq_return']], on='ds')
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df.dropna(inplace=True)

    # 4. Entrenar modelo
    regs = ['stoch_k','stoch_d','qqq_close','qqq_return','y_lag1','y_lag2']
    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds','y'] + regs])

    # 5. Forecast a 30 días hábiles
    future = model.make_future_dataframe(periods=30, freq='B').set_index('ds')
    hist = df.set_index('ds').reindex(future.index)
    win = 5
    future['qqq_close']  = hist['qqq_close'].ffill().fillna(df['qqq_close'].rolling(win).mean().iloc[-1])
    future['qqq_return'] = hist['qqq_return'].ffill().fillna(df['qqq_return'].rolling(win).mean().iloc[-1])
    future['y_lag1']     = hist['y'].shift(1).fillna(df['y'].iloc[-1])
    future['y_lag2']     = hist['y'].shift(2).fillna(df['y'].iloc[-2])
    for c in ['stoch_k','stoch_d']:
        future[c] = hist[c].fillna(df[c].iloc[-1])
    future = future.reset_index()
    forecast = model.predict(future)

    # 6. Unir predicción + reales
    df_pred = forecast[['ds','yhat','yhat_lower','yhat_upper']].merge(
        df[['ds','y']], on='ds', how='inner')
    df_pred['residual'] = df_pred['y'] - df_pred['yhat']
    df_pred['vol_ewma'] = df_pred['residual'].ewm(span=30).std()
    df_pred['std_resid'] = df_pred['residual'] / df_pred['vol_ewma']

    # 7. Calibrar z para IC 95%
    z_95 = np.nanpercentile(np.abs(df_pred['std_resid']), 95)

    # 8. Backtest 90 días → métricas
    cutoff = df['ds'].max() - pd.Timedelta(days=90)
    train = df[df['ds'] < cutoff]
    test  = df[df['ds'] >= cutoff]
    model_bt = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model_bt.add_regressor(r)
    model_bt.fit(train[['ds','y'] + regs])
    fc_bt = model_bt.predict(test[['ds'] + regs])
    bt = fc_bt[['ds','yhat']].merge(test[['ds','y']], on='ds')
    y_true, y_pred = bt['y'], bt['yhat']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100
    r2   = r2_score(y_true, y_pred)

    # 9. Cobertura IC estático
    coverage = ((df_pred['y'] >= df_pred['yhat_lower']) &
                (df_pred['y'] <= df_pred['yhat_upper'])).mean()

    # 10. Precisión direccional
    df_perf = df_pred.copy()
    df_perf['actual_dir'] = df_perf['y'].diff()
    df_perf['pred_dir']   = df_perf['yhat'] - df_perf['y'].shift(1)
    directional_acc = (np.sign(df_perf['actual_dir']) == np.sign(df_perf['pred_dir'])).dropna().mean()

    # 11. Predicción siguiente día hábil + últimos 5 días
    today = pd.to_datetime(datetime.now().date())
    target = today if today.weekday() < 5 else today + BDay(1)
    last5 = df[df['ds'] < target].sort_values('ds').tail(5)

    future_t = pd.DataFrame({'ds': [target]})
    last_row = df[df['ds'] < target].iloc[-1]
    for r in regs:
        future_t[r] = last_row[r]
    pred_t = model.predict(future_t)[['ds','yhat']]
    pred_t = pred_t.merge(df_pred[['ds','vol_ewma']], on='ds', how='left')
    pred_t['vol_ewma'] = pred_t['vol_ewma'].fillna(df_pred['vol_ewma'].iloc[-1])
    pred_t['lower_95'] = pred_t['yhat'] - z_95 * pred_t['vol_ewma']
    pred_t['upper_95'] = pred_t['yhat'] + z_95 * pred_t['vol_ewma']

    # 12. recent_preds para gráfico reciente
    recent_preds = pd.concat([
        forecast[forecast['ds'].isin(last5['ds'])][['ds','yhat','yhat_lower','yhat_upper']],
        pred_t[['ds','yhat','lower_95','upper_95']]
    ], ignore_index=True).sort_values('ds')

    # 13. Gráfico completo
    buf1 = BytesIO()
    plt.figure(figsize=(14,6))
    plt.plot(df['ds'], df['y'], 'k.', alpha=0.6, label='Real')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicción')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, label='IC 95%')
    plt.legend(); plt.tight_layout()
    plt.savefig(buf1, format='png'); plt.close(); buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode()

    # 14. Gráfico recientes con errorbars para todos los puntos
    buf2 = BytesIO()
    plt.figure(figsize=(10,6))
    # Series real
    plt.plot(last5['ds'], last5['y'], 'o-', label='Real', color='#1f77b4')
    # Predicción con líneas y puntos
    plt.plot(recent_preds['ds'], recent_preds['yhat'], '--', label='Predicción', color='#ff7f0e')
    plt.scatter(recent_preds['ds'], recent_preds['yhat'], color='#ff7f0e')
    # Etiquetas de valor
    for x, y in zip(recent_preds['ds'], recent_preds['yhat']):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9, fontfamily='sans-serif')
    # Barras de error para cada punto
    err_lower = recent_preds['yhat'] - recent_preds['lower_95']
    err_upper = recent_preds['upper_95'] - recent_preds['yhat']
    plt.errorbar(
        recent_preds['ds'], recent_preds['yhat'],
        yerr=[err_lower, err_upper],
        fmt='none', ecolor='#2ca02c', capsize=5, label='IC dinámico 95%'
    )
    plt.legend(); plt.tight_layout()
    plt.savefig(buf2, format='png'); plt.close(); buf2.seek(0)
    img2 = base64.b64encode(buf2.read()).decode()

    # 15. Métricas finales sin gráficos de pie
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'coverage': coverage*100,
        'directional_acc': directional_acc*100,
        'predicted_price': float(pred_t.at[0, 'yhat']),
        'lower_95': float(pred_t.at[0, 'lower_95']),
        'upper_95': float(pred_t.at[0, 'upper_95']),
        'target_date': target.date(),
    }

    return {
        'metrics': metrics,
        'plot_full': img1,
        'plot_recent': img2
    }
