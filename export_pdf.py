import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from prophet import Prophet
from ta.momentum import StochasticOscillator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from datetime import datetime
from pandas.tseries.offsets import BDay

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'export'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def entrenar_y_predecir():
    tqqq = (
        pd.read_csv(DATA_DIR / 'TQQQ_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'TQQQ.Close': 'y'})
    )
    stoch = StochasticOscillator(
        close=tqqq['y'], high=tqqq['TQQQ.High'], low=tqqq['TQQQ.Low'],
        window=14, smooth_window=3
    )
    tqqq['stoch_k'] = stoch.stoch()
    tqqq['stoch_d'] = stoch.stoch_signal()
    tqqq['volume_scaled'] = StandardScaler().fit_transform(tqqq[['TQQQ.Volume']])

    qqq = (
        pd.read_csv(DATA_DIR / 'QQQ_data.csv', parse_dates=['Date'])
        .rename(columns={'Date': 'ds', 'QQQ.Close': 'qqq_close'})
    )
    qqq['qqq_return'] = np.log(qqq['qqq_close'] / qqq['qqq_close'].shift(1))

    df = pd.merge(tqqq, qqq[['ds', 'qqq_close', 'qqq_return']], on='ds')
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df.dropna(inplace=True)

    regs = ['stoch_k','stoch_d','qqq_close','qqq_return','y_lag1','y_lag2']
    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    for r in regs:
        model.add_regressor(r)
    model.fit(df[['ds','y'] + regs])

    future = model.make_future_dataframe(periods=1, freq='B').set_index('ds')
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

    df_pred = model.predict(df[['ds'] + regs])
    df_pred = df_pred[['ds', 'yhat']].merge(df[['ds', 'y']], on='ds')
    df_pred['residual'] = df_pred['y'] - df_pred['yhat']
    df_pred['vol_ewma'] = df_pred['residual'].ewm(span=30).std()
    df_pred['std_resid'] = df_pred['residual'] / df_pred['vol_ewma']
    z_95 = np.nanpercentile(np.abs(df_pred['std_resid']), 95)

    vol_ewma_actual = df_pred['vol_ewma'].iloc[-1]
    yhat_pred = forecast.at[0, 'yhat']
    lower_95 = yhat_pred - z_95 * vol_ewma_actual
    upper_95 = yhat_pred + z_95 * vol_ewma_actual

    today = pd.to_datetime(datetime.now().date())
    target = today if today.weekday() < 5 else today + BDay(1)
    last5 = df[df['ds'] < target].sort_values('ds').tail(5)
    recent = pd.concat([
        last5[['ds', 'y']].rename(columns={'y': 'yhat'}),
        pd.DataFrame({'ds': [target], 'yhat': [yhat_pred]})
    ])

    return target, yhat_pred, lower_95, upper_95, recent

def generar_grafico(data: pd.DataFrame, path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(data['ds'], data['yhat'], marker='o', linestyle='--', label='Predicción')
    for x, y in zip(data['ds'], data['yhat']):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
    plt.title('Últimos 5 días y predicción siguiente día')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(path)
    plt.close()

def generar_pdf(path_img: Path, path_pdf: Path, fecha, prediccion, lower, upper):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', 'B', 16)
    pdf.set_text_color(0, 51, 102)  # Azul oscuro
    pdf.cell(0, 10, f'Informe Diario - TQQQ', ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('Times', '', 12)
    pdf.set_text_color(0, 0, 0)  # Negro
    pdf.cell(0, 10, f'Fecha de predicción: {fecha.strftime("%Y-%m-%d")}', ln=True)
    pdf.cell(0, 10, f'Precio esperado: {prediccion:.2f}', ln=True)
    pdf.cell(0, 10, f'Intervalo de confianza 95%: [{lower:.2f}, {upper:.2f}]', ln=True)
    pdf.ln(10)
    pdf.image(str(path_img), x=10, w=180)
    pdf.output(str(path_pdf))

# Ejecutar
fecha, pred, lower, upper, df_grafico = entrenar_y_predecir()
path_img = OUTPUT_DIR / 'grafico_reciente.png'
path_pdf = OUTPUT_DIR / f'reporte_{fecha.date()}.pdf'
generar_grafico(df_grafico, path_img)
generar_pdf(path_img, path_pdf, fecha, pred, lower, upper)
print(f"✅ PDF generado: {path_pdf}")
