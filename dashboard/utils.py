
import pandas as pd
import numpy as np
import matplotlib
# Forzar backend no-GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from prophet import Prophet
from ta.momentum import StochasticOscillator, RSIIndicator, ROCIndicator
from ta.trend import CCIIndicator, MACD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from pandas.tseries.offsets import BDay
from functools import lru_cache
from pathlib import Path
from django.conf import settings
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import CCIIndicator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from ta.volatility import AverageTrueRange



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
    model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)

    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds','y'] + regs])

    # 5. Forecast a 7 días hábiles
    future = model.make_future_dataframe(periods=7, freq='B').set_index('ds')
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

@lru_cache(maxsize=1)
def get_tqqq_signal():
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, CCIIndicator

    # Cargar datos históricos
    tqqq_df = pd.read_csv(BASE_DIR / 'data' / 'TQQQ_data.csv', parse_dates=['Date'])
    tqqq_df.sort_values('Date', inplace=True)
    tqqq_df = tqqq_df.rename(columns={'Date': 'ds', 'TQQQ.Close': 'close'})

    # Calcular indicadores técnicos
    rsi = RSIIndicator(close=tqqq_df['close'], window=14).rsi()
    macd = MACD(close=tqqq_df['close']).macd_diff()
    cci = CCIIndicator(
        high=tqqq_df['TQQQ.High'],
        low=tqqq_df['TQQQ.Low'],
        close=tqqq_df['close'],
        window=20
    ).cci()

    rsi_value = rsi.iloc[-1]
    macd_value = macd.iloc[-1]
    cci_value = cci.iloc[-1]

    # Señal técnica binaria (1 = favorable, 0 = desfavorable)
    rsi_score = 1 if 30 < rsi_value < 60 else 0
    macd_score = 1 if macd_value > 0 else 0
    cci_score = 1 if cci_value > 0 else 0

    # Señal de Prophet
    pred = run_prophet_and_plot()['metrics']['predicted_price']
    last_close = tqqq_df['close'].iloc[-1]
    prophet_score = 1 if pred > last_close else 0

    # Cálculo del puntaje total ponderado
    score = (
        0.25 * rsi_score +
        0.35 * macd_score +
        0.25 * cci_score +
        0.15 * prophet_score
    )

    # Umbrales de decisión
    if score > 0.7:
        decision = 'entrada'
        color = 'success'
        recomendacion = 'Posible entrada'
    elif score < 0.3:
        decision = 'evitar'
        color = 'danger'
        recomendacion = 'Evitar entrada'
    else:
        decision = 'esperar'
        color = 'warning'
        recomendacion = 'Esperar'

    # Dirección visual esperada
    direction = '↑ Subida' if prophet_score == 1 else '↓ Bajada'
    prev_close = tqqq_df['close'].iloc[-2]
    consistency = (np.sign(pred - last_close) == np.sign(last_close - prev_close))

    return {
        'rsi': round(rsi_value, 2),
        'rsi_zone': 'Sobreventa' if rsi_value < 30 else 'Operable' if rsi_value < 60 else 'Sobrecompra',
        'macd': round(macd_value, 2),
        'cci': round(cci_value, 2),
        'direction': direction,
        'consistency': consistency,
        'recomendacion': recomendacion,
        'color': color,
        'ponderado': round(score, 2)
    }



@lru_cache(maxsize=1)
def run_upro_prophet_and_plot():
    # 1. Cargar y preparar UPRO
    upro = (
        pd.read_csv(BASE_DIR / 'data' / 'UPRO_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'UPRO.Close': 'y'})
    )
    stoch = StochasticOscillator(
        close=upro['y'], high=upro['UPRO.High'], low=upro['UPRO.Low'],
        window=14, smooth_window=3
    )
    upro['stoch_k'] = stoch.stoch()
    upro['stoch_d'] = stoch.stoch_signal()
    scaler = StandardScaler()
    upro['volume_scaled'] = scaler.fit_transform(upro[['UPRO.Volume']])

    # 2. Cargar y preparar SPY
    spy = (
        pd.read_csv(BASE_DIR / 'data' / 'SPY_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'SPY.Close': 'spy_close'})
    )
    spy['spy_return'] = np.log(spy['spy_close'] / spy['spy_close'].shift(1))

    # 3. Merge + rezagos
    df = pd.merge(upro, spy[['ds','spy_close','spy_return']], on='ds')
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df.dropna(inplace=True)

    # 4. Entrenar modelo
    regs = ['stoch_k','stoch_d','spy_close','spy_return','y_lag1','y_lag2']
    model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)

    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds','y'] + regs])

    # 5. Forecast a 7 días hábiles
    future = model.make_future_dataframe(periods=7, freq='B').set_index('ds')
    hist = df.set_index('ds').reindex(future.index)
    win = 5
    future['spy_close']  = hist['spy_close'].ffill().fillna(df['spy_close'].rolling(win).mean().iloc[-1])
    future['spy_return'] = hist['spy_return'].ffill().fillna(df['spy_return'].rolling(win).mean().iloc[-1])
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

    # 8. Backtest 90 días
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

    # 9. Cobertura IC
    coverage = ((df_pred['y'] >= df_pred['yhat_lower']) & 
                (df_pred['y'] <= df_pred['yhat_upper'])).mean()

    # 10. Precisión direccional
    df_perf = df_pred.copy()
    df_perf['actual_dir'] = df_perf['y'].diff()
    df_perf['pred_dir']   = df_perf['yhat'] - df_perf['y'].shift(1)
    directional_acc = (np.sign(df_perf['actual_dir']) == np.sign(df_perf['pred_dir'])).dropna().mean()

    # 11. Predicción día hábil siguiente
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

    # 12. recent_preds
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

    # 14. Gráfico recientes
    buf2 = BytesIO()
    plt.figure(figsize=(10,6))
    plt.plot(last5['ds'], last5['y'], 'o-', label='Real', color='#1f77b4')
    plt.plot(recent_preds['ds'], recent_preds['yhat'], '--', label='Predicción', color='#ff7f0e')
    plt.scatter(recent_preds['ds'], recent_preds['yhat'], color='#ff7f0e')
    for x, y in zip(recent_preds['ds'], recent_preds['yhat']):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9, fontfamily='sans-serif')
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

    # 15. Métricas finales
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

@lru_cache(maxsize=1)
def get_upro_signal():
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, CCIIndicator

    # 1. Cargar y ordenar datos
    upro = pd.read_csv(BASE_DIR / 'data' / 'UPRO_data.csv', parse_dates=['Date'])
    upro.sort_values('Date', inplace=True)

    # 2. Calcular indicadores técnicos
    rsi = RSIIndicator(close=upro['UPRO.Close'], window=14).rsi()
    macd = MACD(close=upro['UPRO.Close']).macd_diff()
    cci = CCIIndicator(
        high=upro['UPRO.High'],
        low=upro['UPRO.Low'],
        close=upro['UPRO.Close'],
        window=20
    ).cci()

    rsi_value = rsi.iloc[-1]
    macd_value = macd.iloc[-1]
    cci_value = cci.iloc[-1]

    # 3. Señales binarias
    rsi_score = 1 if 30 < rsi_value < 60 else 0
    macd_score = 1 if macd_value > 0 else 0
    cci_score = 1 if cci_value > 0 else 0

    # 4. Dirección esperada según Prophet
    data = run_upro_prophet_and_plot()
    pred = data['metrics']['predicted_price']
    last_close = upro['UPRO.Close'].iloc[-1]
    prophet_score = 1 if pred > last_close else 0

    direction = '↑ Subida' if prophet_score == 1 else '↓ Bajada'
    prev_close = upro['UPRO.Close'].iloc[-2]
    consistency = (np.sign(pred - last_close) == np.sign(last_close - prev_close))

    # 5. Puntaje ponderado
    score = (
        0.25 * rsi_score +
        0.35 * macd_score +
        0.25 * cci_score +
        0.15 * prophet_score
    )

    # 6. Decisión final
    if score > 0.7:
        recomendacion = 'Posible entrada'
        color = 'success'
    elif score < 0.3:
        recomendacion = 'Evitar entrada'
        color = 'danger'
    else:
        recomendacion = 'Esperar'
        color = 'warning'

    # 7. Zona RSI
    rsi_zone = 'Sobreventa' if rsi_value < 30 else 'Operable' if rsi_value < 60 else 'Sobrecompra'

    return {
        'recomendacion': recomendacion,
        'rsi': round(rsi_value, 2),
        'rsi_zone': rsi_zone,
        'macd': round(macd_value, 2),
        'cci': round(cci_value, 2),
        'direction': direction,
        'consistency': consistency,
        'color': color,
        'ponderado': round(score, 2)
    }




@lru_cache(maxsize=1)
def run_soxl_prophet_and_plot():
    # 1. Cargar y preparar SOXL
    soxl = (
        pd.read_csv(BASE_DIR / 'data' / 'SOXL_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'SOXL.Close': 'y'})
    )
    stoch = StochasticOscillator(
        close=soxl['y'], high=soxl['SOXL.High'], low=soxl['SOXL.Low'],
        window=14, smooth_window=3
    )
    soxl['stoch_k'] = stoch.stoch()
    soxl['stoch_d'] = stoch.stoch_signal()
    scaler = StandardScaler()
    soxl['volume_scaled'] = scaler.fit_transform(soxl[['SOXL.Volume']])

    # 2. Rezagos
    soxl['y_lag1'] = soxl['y'].shift(1)
    soxl['y_lag2'] = soxl['y'].shift(2)
    soxl.dropna(inplace=True)

    # 3. Entrenar modelo Prophet
    regs = ['stoch_k', 'stoch_d', 'volume_scaled', 'y_lag1', 'y_lag2']
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    for r in regs:
        model.add_regressor(r)
    model.fit(soxl[['ds','y'] + regs])

    # 4. Forecast 7 días hábiles
    future = model.make_future_dataframe(periods=7, freq='B').set_index('ds')
    hist = soxl.set_index('ds').reindex(future.index)
    win = 5
    future['volume_scaled'] = hist['volume_scaled'].ffill().fillna(soxl['volume_scaled'].rolling(win).mean().iloc[-1])
    future['y_lag1'] = hist['y'].shift(1).fillna(soxl['y'].iloc[-1])
    future['y_lag2'] = hist['y'].shift(2).fillna(soxl['y'].iloc[-2])
    for c in ['stoch_k','stoch_d']:
        future[c] = hist[c].fillna(soxl[c].iloc[-1])
    future = future.reset_index()
    forecast = model.predict(future)

    # 5. Combinar predicción con reales
    df_pred = forecast[['ds','yhat','yhat_lower','yhat_upper']].merge(
        soxl[['ds','y']], on='ds', how='inner')
    df_pred['residual'] = df_pred['y'] - df_pred['yhat']
    df_pred['vol_ewma'] = df_pred['residual'].ewm(span=30).std()
    df_pred['std_resid'] = df_pred['residual'] / df_pred['vol_ewma']

    # 6. Calibrar z para IC
    z_95 = np.nanpercentile(np.abs(df_pred['std_resid']), 95)

    # 7. Backtest 90 días
    cutoff = soxl['ds'].max() - pd.Timedelta(days=90)
    train = soxl[soxl['ds'] < cutoff]
    test = soxl[soxl['ds'] >= cutoff]
    model_bt = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model_bt.add_regressor(r)
    model_bt.fit(train[['ds','y'] + regs])
    fc_bt = model_bt.predict(test[['ds'] + regs])
    bt = fc_bt[['ds','yhat']].merge(test[['ds','y']], on='ds')
    rmse = np.sqrt(mean_squared_error(bt['y'], bt['yhat']))
    mae = mean_absolute_error(bt['y'], bt['yhat'])
    mape = (np.abs((bt['y'] - bt['yhat']) / bt['y']).mean()) * 100
    r2 = r2_score(bt['y'], bt['yhat'])

    # 8. Cobertura IC y precisión direccional
    coverage = ((df_pred['y'] >= df_pred['yhat_lower']) & (df_pred['y'] <= df_pred['yhat_upper'])).mean()
    df_perf = df_pred.copy()
    df_perf['actual_dir'] = df_perf['y'].diff()
    df_perf['pred_dir'] = df_perf['yhat'] - df_perf['y'].shift(1)
    directional_acc = (np.sign(df_perf['actual_dir']) == np.sign(df_perf['pred_dir'])).dropna().mean()

    # 9. Predicción siguiente día hábil
    today = pd.to_datetime(datetime.now().date())
    target = today if today.weekday() < 5 else today + BDay(1)
    last5 = soxl[soxl['ds'] < target].sort_values('ds').tail(5)
    future_t = pd.DataFrame({'ds': [target]})
    last_row = soxl[soxl['ds'] < target].iloc[-1]
    for r in regs:
        future_t[r] = last_row[r]
    pred_t = model.predict(future_t)[['ds','yhat']]
    pred_t = pred_t.merge(df_pred[['ds','vol_ewma']], on='ds', how='left')
    pred_t['vol_ewma'] = pred_t['vol_ewma'].fillna(df_pred['vol_ewma'].iloc[-1])
    pred_t['lower_95'] = pred_t['yhat'] - z_95 * pred_t['vol_ewma']
    pred_t['upper_95'] = pred_t['yhat'] + z_95 * pred_t['vol_ewma']

    # 10. Graficar
    buf1 = BytesIO()
    plt.figure(figsize=(14,6))
    plt.plot(soxl['ds'], soxl['y'], 'k.', alpha=0.6, label='Real')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicción')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, label='IC 95%')
    plt.legend(); plt.tight_layout()
    plt.savefig(buf1, format='png'); plt.close(); buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode()

    buf2 = BytesIO()
    plt.figure(figsize=(10,6))
    plt.plot(last5['ds'], last5['y'], 'o-', label='Real', color='#1f77b4')
    plt.plot([*last5['ds'], target], [*last5['y'].values, pred_t['yhat'].iloc[0]], '--', color='#ff7f0e', label='Predicción')
    plt.scatter(target, pred_t['yhat'], color='#ff7f0e')
    yhat_val = pred_t['yhat'].values[0]
    err_low = (yhat_val - pred_t['lower_95'].values[0])
    err_up = (pred_t['upper_95'].values[0] - yhat_val)
    plt.errorbar([target], [yhat_val], yerr=[[err_low], [err_up]], fmt='none', ecolor='#2ca02c', capsize=5, label='IC dinámico 95%')
    plt.legend(); plt.tight_layout()
    plt.savefig(buf2, format='png'); plt.close(); buf2.seek(0)
    img2 = base64.b64encode(buf2.read()).decode()

    # 11. Métricas
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

@lru_cache(maxsize=1)
def get_soxl_signal():
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, CCIIndicator

    # 1. Obtener predicción de Prophet para SOXL
    output = run_soxl_prophet_and_plot()
    pred = output['metrics']['predicted_price']

    # 2. Cargar y preparar datos
    df = pd.read_csv(BASE_DIR / 'data' / 'SOXL_data.csv', parse_dates=['Date'])
    df = df.rename(columns={'Date': 'ds', 'SOXL.Close': 'close'})
    df.sort_values('ds', inplace=True)

    # 3. Calcular indicadores
    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close']).macd_diff()
    cci = CCIIndicator(
        high=df['SOXL.High'],
        low=df['SOXL.Low'],
        close=df['close'],
        window=20
    ).cci()

    rsi_value = rsi.iloc[-1]
    macd_value = macd.iloc[-1]
    cci_value = cci.iloc[-1]

    # 4. Señales binarias
    rsi_score = 1 if 30 < rsi_value < 60 else 0
    macd_score = 1 if macd_value > 0 else 0
    cci_score = 1 if cci_value > 0 else 0

    # 5. Dirección predicha por Prophet
    last_close = df['close'].iloc[-1]
    prophet_score = 1 if pred > last_close else 0
    direction = '↑ Subida' if prophet_score == 1 else '↓ Bajada'

    # 6. Consistencia direccional con ayer
    prev_close = df['close'].iloc[-2]
    consistency = (np.sign(pred - last_close) == np.sign(last_close - prev_close))

    # 7. Puntaje ponderado
    score = (
        0.25 * rsi_score +
        0.35 * macd_score +
        0.25 * cci_score +
        0.15 * prophet_score
    )

    # 8. Decisión final
    if score > 0.7:
        recomendacion = 'Posible entrada'
        color = 'success'
    elif score < 0.3:
        recomendacion = 'Evitar entrada'
        color = 'danger'
    else:
        recomendacion = 'Esperar'
        color = 'warning'

    # 9. Zona RSI
    rsi_zone = 'Sobreventa' if rsi_value < 30 else 'Operable' if rsi_value < 60 else 'Sobrecompra'

    return {
        'rsi': round(rsi_value, 2),
        'rsi_zone': rsi_zone,
        'macd': round(macd_value, 2),
        'cci': round(cci_value, 2),
        'direction': direction,
        'consistency': consistency,
        'recomendacion': recomendacion,
        'color': color,
        'ponderado': round(score, 2)
    }

@lru_cache(maxsize=1)
def run_qqq_prophet_and_plot():
    # 1. Cargar QQQ
    qqq = (
        pd.read_csv(BASE_DIR / 'data' / 'QQQ_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'QQQ.Close': 'y'})
    )
    qqq['qqq_return'] = np.log(qqq['y'] / qqq['y'].shift(1))
    qqq['y_lag1'] = qqq['y'].shift(1)
    qqq['y_lag2'] = qqq['y'].shift(2)

    stoch = StochasticOscillator(close=qqq['y'], high=qqq['QQQ.High'], low=qqq['QQQ.Low'])
    qqq['stoch_k'] = stoch.stoch()
    qqq['stoch_d'] = stoch.stoch_signal()

    # 2. Cargar TQQQ y preparar regresores
    tqqq = pd.read_csv(BASE_DIR / 'data' / 'TQQQ_data.csv', parse_dates=['Date'])
    tqqq = tqqq.rename(columns={'Date': 'ds'})
    tqqq['tqqq_return'] = np.log(tqqq['TQQQ.Close'] / tqqq['TQQQ.Close'].shift(1))
    tqqq['tqqq_lag1'] = tqqq['TQQQ.Close'].shift(1)

    stoch_t = StochasticOscillator(close=tqqq['TQQQ.Close'], high=tqqq['TQQQ.High'], low=tqqq['TQQQ.Low'])
    tqqq['tqqq_stoch_k'] = stoch_t.stoch()
    tqqq['tqqq_stoch_d'] = stoch_t.stoch_signal()

    # 3. Merge
    df = pd.merge(qqq, tqqq[['ds', 'TQQQ.Close', 'tqqq_return', 'tqqq_lag1', 'tqqq_stoch_k', 'tqqq_stoch_d']], on='ds')
    df = df.rename(columns={'TQQQ.Close': 'tqqq_close'})
    df.dropna(inplace=True)

    # 4. Entrenar modelo
    regs = ['qqq_return', 'y_lag1', 'y_lag2', 'stoch_k', 'stoch_d',
            'tqqq_close', 'tqqq_return', 'tqqq_lag1', 'tqqq_stoch_k', 'tqqq_stoch_d']

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds', 'y'] + regs])

    # 5. Forecast
    future = model.make_future_dataframe(periods=7, freq='B').set_index('ds')
    hist = df.set_index('ds').reindex(future.index)
    win = 5
    for col in regs:
        future[col] = hist[col].fillna(df[col].rolling(win).mean().iloc[-1])
    future = future.reset_index()
    forecast = model.predict(future)

    # 6. Unir predicciones
    df_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
        df[['ds', 'y']], on='ds', how='inner'
    )
    df_pred['residual'] = df_pred['y'] - df_pred['yhat']
    df_pred['vol_ewma'] = df_pred['residual'].ewm(span=30).std()
    df_pred['std_resid'] = df_pred['residual'] / df_pred['vol_ewma']
    z_95 = np.nanpercentile(np.abs(df_pred['std_resid']), 95)

    # 7. Backtest
    cutoff = df['ds'].max() - pd.Timedelta(days=90)
    train = df[df['ds'] < cutoff]
    test = df[df['ds'] >= cutoff]

    model_bt = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model_bt.add_regressor(r)
    model_bt.fit(train[['ds', 'y'] + regs])
    fc_bt = model_bt.predict(test[['ds'] + regs])
    bt = fc_bt[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')
    y_true, y_pred = bt['y'], bt['yhat']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100
    r2 = r2_score(y_true, y_pred)

    # 8. Métricas adicionales
    coverage = ((df_pred['y'] >= df_pred['yhat_lower']) &
                (df_pred['y'] <= df_pred['yhat_upper'])).mean()
    df_perf = df_pred.copy()
    df_perf['actual_dir'] = df_perf['y'].diff()
    df_perf['pred_dir'] = df_perf['yhat'] - df_perf['y'].shift(1)
    directional_acc = (np.sign(df_perf['actual_dir']) == np.sign(df_perf['pred_dir'])).dropna().mean()

    # 9. Predicción siguiente día hábil
    today = pd.to_datetime(datetime.now().date())
    target = today if today.weekday() < 5 else today + BDay(1)
    last5 = df[df['ds'] < target].sort_values('ds').tail(5)
    last_row = df[df['ds'] < target].iloc[-1]
    future_t = pd.DataFrame({'ds': [target]})
    for r in regs:
        future_t[r] = last_row[r]
    pred_t = model.predict(future_t)[['ds', 'yhat']]
    pred_t = pred_t.merge(df_pred[['ds', 'vol_ewma']], on='ds', how='left')
    pred_t['vol_ewma'] = pred_t['vol_ewma'].fillna(df_pred['vol_ewma'].iloc[-1])
    pred_t['lower_95'] = pred_t['yhat'] - z_95 * pred_t['vol_ewma']
    pred_t['upper_95'] = pred_t['yhat'] + z_95 * pred_t['vol_ewma']

    # 10. Gráficos
    buf1 = BytesIO()
    plt.figure(figsize=(14, 6))
    plt.plot(df['ds'], df['y'], 'k.', alpha=0.6, label='Real')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicción')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, label='IC 95%')
    plt.legend(); plt.tight_layout()
    plt.savefig(buf1, format='png'); plt.close(); buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode()

    recent_preds = pd.concat([
        forecast[forecast['ds'].isin(last5['ds'])][['ds', 'yhat']],
        pred_t[['ds', 'yhat']]
    ], ignore_index=True).sort_values('ds')
    buf2 = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(last5['ds'], last5['y'], 'o-', label='Real', color='#1f77b4')
    plt.plot(recent_preds['ds'], recent_preds['yhat'], '--', label='Predicción', color='#ff7f0e')
    plt.scatter(recent_preds['ds'], recent_preds['yhat'], color='#ff7f0e')
    for x, y in zip(recent_preds['ds'], recent_preds['yhat']):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
    plt.legend(); plt.tight_layout()
    plt.savefig(buf2, format='png'); plt.close(); buf2.seek(0)
    img2 = base64.b64encode(buf2.read()).decode()

    return {
        'plot_full': img1,
        'plot_recent': img2,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'coverage': coverage * 100,
            'directional_acc': directional_acc * 100,
            'predicted_price': float(pred_t.at[0, 'yhat']),
            'lower_95': float(pred_t.at[0, 'lower_95']),
            'upper_95': float(pred_t.at[0, 'upper_95']),
            'target_date': target.date()
        }
    }


@lru_cache(maxsize=1)
def get_qqq_signal():
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, CCIIndicator

    df = pd.read_csv(BASE_DIR / 'data' / 'QQQ_data.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df = df.rename(columns={'Date': 'ds', 'QQQ.Close': 'close'})

    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close']).macd_diff()
    cci = CCIIndicator(high=df['QQQ.High'], low=df['QQQ.Low'], close=df['close'], window=20).cci()

    rsi_value = rsi.iloc[-1]
    macd_value = macd.iloc[-1]
    cci_value = cci.iloc[-1]

    rsi_score = 1 if 30 < rsi_value < 60 else 0
    macd_score = 1 if macd_value > 0 else 0
    cci_score = 1 if cci_value > 0 else 0

    pred = run_qqq_prophet_and_plot()['metrics']['predicted_price']
    last_close = df['close'].iloc[-1]
    prophet_score = 1 if pred > last_close else 0

    score = (
        0.25 * rsi_score +
        0.35 * macd_score +
        0.25 * cci_score +
        0.15 * prophet_score
    )

    if score > 0.7:
        recomendacion = 'Posible entrada'
        color = 'success'
    elif score < 0.3:
        recomendacion = 'Evitar entrada'
        color = 'danger'
    else:
        recomendacion = 'Esperar'
        color = 'warning'

    direction = '↑ Subida' if prophet_score == 1 else '↓ Bajada'
    prev_close = df['close'].iloc[-2]
    consistency = (np.sign(pred - last_close) == np.sign(last_close - prev_close))

    return {
        'rsi': round(rsi_value, 2),
        'rsi_zone': 'Sobreventa' if rsi_value < 30 else 'Operable' if rsi_value < 60 else 'Sobrecompra',
        'macd': round(macd_value, 2),
        'cci': round(cci_value, 2),
        'direction': direction,
        'consistency': consistency,
        'recomendacion': recomendacion,
        'color': color,
        'ponderado': round(score, 2)
    }


@lru_cache(maxsize=1)
def run_rhhby_prophet_and_plot():
    # Cargar RHHBY
    df = pd.read_csv(BASE_DIR / 'data' / 'RHHBY_data.csv', parse_dates=['Date'])
    df = df.rename(columns={'Date': 'ds', 'RHHBY.Close': 'y'})
    df.sort_values('ds', inplace=True)

    # Indicadores RHHBY (sin técnicos)
    df['rhhby_return'] = np.log(df['y'] / df['y'].shift(1))
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df['log_y'] = np.log(df['y'])

    # Cargar XLV
    xlv = pd.read_csv(BASE_DIR / 'data' / 'XLV_data.csv', parse_dates=['Date'])
    xlv = xlv.rename(columns={'Date': 'ds'})
    xlv['xlv_return'] = np.log(xlv['XLV.Close'] / xlv['XLV.Close'].shift(1))
    xlv['xlv_lag1'] = xlv['XLV.Close'].shift(1)

    # Merge
    df = pd.merge(df, xlv[['ds', 'XLV.Close', 'xlv_return', 'xlv_lag1']], on='ds')
    df = df.rename(columns={'XLV.Close': 'xlv_close'})
    df.dropna(inplace=True)

    # Regresores
    regs = ['rhhby_return', 'y_lag1', 'y_lag2', 'xlv_close', 'xlv_return', 'xlv_lag1']

    # Prophet
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds', 'log_y'] + regs].rename(columns={'log_y': 'y'}))

    # Forecast
    future = model.make_future_dataframe(periods=7, freq='B').set_index('ds')
    hist = df.set_index('ds').reindex(future.index)
    win = 5
    for col in regs:
        future[col] = hist[col].fillna(df[col].rolling(win).mean().iloc[-1])
    future = future.reset_index()
    forecast = model.predict(future)
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

    df_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
        df[['ds', 'y']], on='ds', how='inner'
    )
    df_pred['residual'] = df_pred['y'] - df_pred['yhat']
    df_pred['vol_ewma'] = df_pred['residual'].ewm(span=30).std()
    df_pred['std_resid'] = df_pred['residual'] / df_pred['vol_ewma']
    z_95 = np.nanpercentile(np.abs(df_pred['std_resid']), 95)

    cutoff = df['ds'].max() - pd.Timedelta(days=90)
    train = df[df['ds'] < cutoff]
    test = df[df['ds'] >= cutoff]

    model_bt = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model_bt.add_regressor(r)
    model_bt.fit(train[['ds', 'log_y'] + regs].rename(columns={'log_y': 'y'}))
    fc_bt = model_bt.predict(test[['ds'] + regs])
    fc_bt['yhat'] = np.exp(fc_bt['yhat'])
    bt = fc_bt[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')
    y_true, y_pred = bt['y'], bt['yhat']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100
    r2 = r2_score(y_true, y_pred)

    coverage = ((df_pred['y'] >= df_pred['yhat_lower']) & (df_pred['y'] <= df_pred['yhat_upper'])).mean()
    df_perf = df_pred.copy()
    df_perf['actual_dir'] = df_perf['y'].diff()
    df_perf['pred_dir'] = df_perf['yhat'] - df_perf['y'].shift(1)
    directional_acc = (np.sign(df_perf['actual_dir']) == np.sign(df_perf['pred_dir'])).dropna().mean()

    today = pd.to_datetime(datetime.now().date())
    target = today if today.weekday() < 5 else today + BDay(1)
    last5 = df[df['ds'] < target].sort_values('ds').tail(5)
    last_row = df[df['ds'] < target].iloc[-1]
    future_t = pd.DataFrame({'ds': [target]})
    for r in regs:
        future_t[r] = last_row[r]
    pred_t = model.predict(future_t)[['ds', 'yhat']]
    pred_t['yhat'] = np.exp(pred_t['yhat'])
    pred_t = pred_t.merge(df_pred[['ds', 'vol_ewma']], on='ds', how='left')
    pred_t['vol_ewma'] = pred_t['vol_ewma'].fillna(df_pred['vol_ewma'].iloc[-1])
    pred_t['lower_95'] = pred_t['yhat'] - z_95 * pred_t['vol_ewma']
    pred_t['upper_95'] = pred_t['yhat'] + z_95 * pred_t['vol_ewma']

    buf1 = BytesIO()
    plt.figure(figsize=(14, 6))
    plt.plot(df['ds'], df['y'], 'k.', alpha=0.6, label='Real')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicción')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, label='IC 95%')
    plt.legend(); plt.tight_layout()
    plt.savefig(buf1, format='png'); plt.close(); buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode()

    recent_preds = pd.concat([
        forecast[forecast['ds'].isin(last5['ds'])][['ds', 'yhat']],
        pred_t[['ds', 'yhat']]
    ], ignore_index=True).sort_values('ds')
    buf2 = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(last5['ds'], last5['y'], 'o-', label='Real', color='#1f77b4')
    plt.plot(recent_preds['ds'], recent_preds['yhat'], '--', label='Predicción', color='#ff7f0e')
    plt.scatter(recent_preds['ds'], recent_preds['yhat'], color='#ff7f0e')
    for x, y in zip(recent_preds['ds'], recent_preds['yhat']):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
    plt.legend(); plt.tight_layout()
    plt.savefig(buf2, format='png'); plt.close(); buf2.seek(0)
    img2 = base64.b64encode(buf2.read()).decode()

    return {
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'coverage': coverage * 100,
            'directional_acc': directional_acc * 100,
            'predicted_price': float(pred_t.at[0, 'yhat']),
            'lower_95': float(pred_t.at[0, 'lower_95']),
            'upper_95': float(pred_t.at[0, 'upper_95']),
            'target_date': target.date()
        },
        'plot_full': img1,
        'plot_recent': img2
    }


@lru_cache(maxsize=1)
def get_rhhby_signal():

    # Cargar datos de RHHBY
    df = pd.read_csv(BASE_DIR / 'data' / 'RHHBY_data.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df = df.rename(columns={'Date': 'ds', 'RHHBY.Close': 'close'})

    # Calcular indicadores técnicos
    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close']).macd_diff()
    cci = CCIIndicator(high=df['RHHBY.High'], low=df['RHHBY.Low'], close=df['close'], window=20).cci()

    rsi_value = rsi.iloc[-1]
    macd_value = macd.iloc[-1]
    cci_value = cci.iloc[-1]

    rsi_score = 1 if 30 < rsi_value < 60 else 0
    macd_score = 1 if macd_value > 0 else 0
    cci_score = 1 if cci_value > 0 else 0

    # Obtener predicción Prophet
    pred = run_rhhby_prophet_and_plot()['metrics']['predicted_price']
    last_close = df['close'].iloc[-1]
    prophet_score = 1 if pred > last_close else 0

    score = (
        0.25 * rsi_score +
        0.35 * macd_score +
        0.25 * cci_score +
        0.15 * prophet_score
    )

    if score > 0.7:
        recomendacion = 'Posible entrada'
        color = 'success'
    elif score < 0.3:
        recomendacion = 'Evitar entrada'
        color = 'danger'
    else:
        recomendacion = 'Esperar'
        color = 'warning'

    direction = '↑ Subida' if prophet_score == 1 else '↓ Bajada'
    prev_close = df['close'].iloc[-2]
    consistency = (np.sign(pred - last_close) == np.sign(last_close - prev_close))

    return {
        'rsi': round(rsi_value, 2),
        'rsi_zone': 'Sobreventa' if rsi_value < 30 else 'Operable' if rsi_value < 60 else 'Sobrecompra',
        'macd': round(macd_value, 2),
        'cci': round(cci_value, 2),
        'direction': direction,
        'consistency': consistency,
        'recomendacion': recomendacion,
        'color': color,
        'ponderado': round(score, 2)
    }
