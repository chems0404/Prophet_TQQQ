from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from prophet import Prophet
from ta.momentum import StochasticOscillator
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pandas.tseries.offsets import BDay

# Rutas
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'export'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Función principal
def exportar_pdf_diario():
    # Cargar y preparar datos
    tqqq = pd.read_csv(DATA_DIR / 'TQQQ_data.csv', parse_dates=['Date']).rename(columns={'Date': 'ds', 'TQQQ.Close': 'y'})
    stoch = StochasticOscillator(close=tqqq['y'], high=tqqq['TQQQ.High'], low=tqqq['TQQQ.Low'], window=14, smooth_window=3)
    tqqq['stoch_k'] = stoch.stoch()
    tqqq['stoch_d'] = stoch.stoch_signal()
    tqqq['volume_scaled'] = StandardScaler().fit_transform(tqqq[['TQQQ.Volume']])

    qqq = pd.read_csv(DATA_DIR / 'QQQ_data.csv', parse_dates=['Date']).rename(columns={'Date': 'ds', 'QQQ.Close': 'qqq_close'})
    qqq['qqq_return'] = np.log(qqq['qqq_close'] / qqq['qqq_close'].shift(1))

    df = pd.merge(tqqq, qqq[['ds','qqq_close','qqq_return']], on='ds')
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df.dropna(inplace=True)

    regs = ['stoch_k','stoch_d','qqq_close','qqq_return','y_lag1','y_lag2']
    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds','y'] + regs])

    # Predicción siguiente día hábil
    today = pd.to_datetime(datetime.now().date())
    target = today if today.weekday() < 5 else today + BDay(1)
    future_t = pd.DataFrame({'ds': [target]})
    last_row = df[df['ds'] < target].iloc[-1]
    for r in regs:
        future_t[r] = last_row[r]
    pred_t = model.predict(future_t)[['ds','yhat']]

    # IC dinámico
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

    df_pred = forecast[['ds','yhat','yhat_lower','yhat_upper']].merge(df[['ds','y']], on='ds', how='inner')
    df_pred['residual'] = df_pred['y'] - df_pred['yhat']
    df_pred['vol_ewma'] = df_pred['residual'].ewm(span=30).std()
    df_pred['std_resid'] = df_pred['residual'] / df_pred['vol_ewma']
    z_95 = np.nanpercentile(np.abs(df_pred['std_resid']), 95)

    pred_t = pred_t.merge(df_pred[['ds','vol_ewma']], on='ds', how='left')
    pred_t['vol_ewma'] = pred_t['vol_ewma'].fillna(df_pred['vol_ewma'].iloc[-1])
    pred_t['lower_95'] = pred_t['yhat'] - z_95 * pred_t['vol_ewma']
    pred_t['upper_95'] = pred_t['yhat'] + z_95 * pred_t['vol_ewma']

    # Últimos 5 reales + predicción
    last5 = df[df['ds'] < target].sort_values('ds').tail(5)
    recent = pd.concat([
        last5[['ds', 'y']].rename(columns={'y': 'yhat'}),
        pred_t[['ds', 'yhat', 'lower_95', 'upper_95']]
    ])

    # Gráfico
    img_path = OUTPUT_DIR / 'grafico_reciente.png'
    plt.figure(figsize=(10, 5))
    plt.plot(recent['ds'], recent['yhat'], marker='o', linestyle='--', label='Predicción', color='#0056b3')
    plt.errorbar(recent['ds'], recent['yhat'],
                 yerr=[recent['yhat'] - recent.get('lower_95', recent['yhat']),
                       recent.get('upper_95', recent['yhat']) - recent['yhat']],
                 fmt='none', ecolor='gray', capsize=5)
    for x, y in zip(recent['ds'], recent['yhat']):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
    plt.title('Últimos 5 días y predicción siguiente día')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(img_path)
    plt.close()

    # PDF
    pdf_path = OUTPUT_DIR / f'reporte_{target.date()}.pdf'
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', 'B', 16)
    pdf.set_text_color(0, 70, 140)  # Azul fuerte
    pdf.cell(0, 10, 'Informe Diario - TQQQ', ln=True, align='C')
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Times', '', 12)
    pdf.cell(0, 10, f'Fecha de predicción: {target.date()}', ln=True)
    pdf.cell(0, 10, f'Precio esperado: {pred_t["yhat"].iloc[0]:.2f}', ln=True)
    pdf.cell(0, 10, f'Intervalo de confianza 95%: [{pred_t["lower_95"].iloc[0]:.2f}, {pred_t["upper_95"].iloc[0]:.2f}]', ln=True)
    pdf.ln(5)
    pdf.image(str(img_path), x=10, w=190)
    pdf.output(str(pdf_path))

    return pdf_path

pdf_path = exportar_pdf_diario()
pdf_path

