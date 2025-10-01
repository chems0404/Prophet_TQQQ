# backtest_semaforo.py
import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.trend import CCIIndicator, MACD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== Config =====
N_DAYS = 90                      # últimos 30 días hábiles (presentes en los datos)
EPS_BAND = 0.001                 # ±0.1% para interpretar "se mantiene" en Prophet
W_RSI = 0.2; W_MACD = 0.2; W_CCI = 0.2; W_PROP = 0.4
TH_LONG = 0.7                    # score > 0.7 -> VERDE (comprar)
TH_SHORT = 0.3                   # score < 0.3 -> ROJO (vender)
FEE_BPS = 0.0                    # comisión por día operado (p.b.)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "backtests_semaforo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utilidades ----------
def last_n_business_days_present(dates: pd.Series, n: int):
    ds = pd.Series(pd.to_datetime(dates.unique())).sort_values()
    biz = ds[ds.dt.dayofweek < 5]
    return list(biz.tail(n))

def decide_direction(delta_rel: float, band: float = EPS_BAND) -> str:
    if np.isnan(delta_rel): return "se mantiene"
    if delta_rel > band:  return "sube"
    if delta_rel < -band: return "baja"
    return "se mantiene"

def metrics_summary(bt: pd.DataFrame) -> dict:
    if len(bt) == 0:
        return dict(mae=np.nan, rmse=np.nan, mape=np.nan, r2=np.nan,
                    directional_acc=np.nan, coverage=np.nan)
    y = bt["y_real"].values
    yp = bt["y_pred"].values
    mae = mean_absolute_error(y, yp)
    rmse = mean_squared_error(y, yp) ** 0.5
    mape = (np.abs((y - yp) / y).mean()) * 100
    r2 = r2_score(y, yp)
    dir_acc = np.nan
    if len(bt) >= 2:
        dir_acc = (np.sign(bt["y_real"].diff()) == np.sign(bt["y_pred"].diff())).iloc[1:].mean() * 100
    coverage = bt["in_ci"].mean() * 100
    return dict(mae=mae, rmse=rmse, mape=mape, r2=r2,
                directional_acc=dir_acc, coverage=coverage)

def simulate_equity(bt_df: pd.DataFrame, initial=1000.0, fee_bps=FEE_BPS):
    """
    Señal semáforo:
      - 'verde'  -> long 100%
      - 'rojo'   -> short 100%
      - 'amarillo' -> cash
    """
    equity = initial
    eq, pos, rets, fees = [], [], [], []
    daily_fee = fee_bps / 10000.0

    for _, row in bt_df.iterrows():
        if row['semaforo'] == 'verde':
            gross_ret = (row['y_real'] / row['prev_close']) - 1.0
            position = 'long'
        elif row['semaforo'] == 'rojo':
            gross_ret = (row['prev_close'] / row['y_real']) - 1.0
            position = 'short'
        else:
            gross_ret = 0.0
            position = 'flat'

        net_ret = gross_ret - (daily_fee if position != 'flat' else 0.0)
        equity *= (1.0 + net_ret)
        eq.append(equity); pos.append(position); rets.append(net_ret)
        fees.append(daily_fee if position != 'flat' else 0.0)

    bt_df['position']  = pos
    bt_df['strat_ret'] = rets
    bt_df['fee']       = fees
    bt_df['equity']    = eq

    total_ret = (equity / initial - 1.0) * 100.0
    return bt_df, equity, total_ret

# ---------- Núcleo Prophet + RF (idéntico a tu pipeline) ----------
def prophet_rf_one_day(train_df: pd.DataFrame, full_df: pd.DataFrame, regs: list[str],
                       target_day: pd.Timestamp, mode: str):
    """
    mode: 'level'  -> entrena Prophet en nivel (TQQQ/UPRO/SOXL/QQQ)
          'logexp' -> entrena en log(y) y expone (RHHBY/BTC)
    train_df: columnas ['ds','y', ...regs...], ordenadas por ds
    full_df:  dataset completo para obtener y_real y prev_close del target_day
    """
    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    for r in regs:
        m.add_regressor(r)

    if mode == "logexp":
        fit_df = train_df.rename(columns={'y': 'y_level'}).copy()
        fit_df['y'] = np.log(fit_df['y_level'])
        m.fit(fit_df[['ds','y'] + regs])
    else:
        m.fit(train_df[['ds','y'] + regs])

    # Preparar el punto objetivo usando los últimos regresores conocidos (misma lógica de tu pipeline)
    last_row = train_df.iloc[-1]
    future_t = pd.DataFrame({'ds': [pd.Timestamp(target_day)]})
    for r in regs:
        future_t[r] = last_row[r]

    base = m.predict(future_t).iloc[0]
    if mode == "logexp":
        yhat_base  = float(np.exp(base['yhat']))
        yhat_lower = float(np.exp(base['yhat_lower']))
        yhat_upper = float(np.exp(base['yhat_upper']))
        hist_fc = m.predict(train_df[['ds'] + regs])
        hist_fc['yhat_level'] = np.exp(hist_fc['yhat'])
        df_pred = hist_fc[['ds','yhat_level']].merge(
            train_df[['ds','y']].rename(columns={'y':'y_level'}), on='ds', how='inner'
        )
        df_pred['residual'] = df_pred['y_level'] - df_pred['yhat_level']
    else:
        yhat_base  = float(base['yhat'])
        yhat_lower = float(base['yhat_lower'])
        yhat_upper = float(base['yhat_upper'])
        hist_fc = m.predict(train_df[['ds'] + regs])
        df_pred = hist_fc[['ds','yhat']].merge(train_df[['ds','y']], on='ds', how='inner')
        df_pred['residual'] = df_pred['y'] - df_pred['yhat']

    X_resid = train_df.set_index('ds').loc[df_pred['ds'], regs]
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    rf.fit(X_resid, df_pred['residual'])

    # Ajuste con RF en el target day
    yhat_adj = yhat_base + float(rf.predict(future_t[regs])[0])

    # Reales y prev_close
    row_target = full_df[full_df['ds'] == pd.Timestamp(target_day)]
    if row_target.empty: 
        return None
    y_real = float(row_target['y'].iloc[0])
    idx = full_df.index[full_df['ds'] == pd.Timestamp(target_day)][0]
    prev_close = float(full_df.iloc[idx-1]['y']) if idx > 0 else np.nan

    # Señales & métricas puntuales
    delta_pred_rel = (yhat_adj - prev_close)/prev_close if np.isfinite(prev_close) and prev_close != 0 else np.nan
    delta_real_rel = (y_real    - prev_close)/prev_close if np.isfinite(prev_close) and prev_close != 0 else np.nan
    direccion_pred = decide_direction(delta_pred_rel, EPS_BAND)
    direccion_real = decide_direction(delta_real_rel, EPS_BAND)
    in_ci = (y_real >= yhat_lower) and (y_real <= yhat_upper)

    return dict(
        prev_close=prev_close, y_real=y_real, y_pred=float(yhat_adj),
        yhat_base=float(yhat_base), yhat_lower=float(yhat_lower), yhat_upper=float(yhat_upper),
        delta_pred_rel=float(delta_pred_rel) if np.isfinite(delta_pred_rel) else np.nan,
        delta_real_rel=float(delta_real_rel) if np.isfinite(delta_real_rel) else np.nan,
        direccion_pred=direccion_pred, direccion_real=direccion_real, in_ci=bool(in_ci)
    )

# ---------- Preparación de datos por activo (misma que tu pipeline) ----------
def prep_TQQQ(dd: Path):
    tqqq = pd.read_csv(dd/'TQQQ_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds','TQQQ.Close':'y'}
    )
    st = StochasticOscillator(close=tqqq['y'], high=tqqq['TQQQ.High'], low=tqqq['TQQQ.Low'], window=14, smooth_window=3)
    tqqq['stoch_k']=st.stoch(); tqqq['stoch_d']=st.stoch_signal()
    scaler = StandardScaler(); tqqq['volume_scaled']=scaler.fit_transform(tqqq[['TQQQ.Volume']])
    qqq = pd.read_csv(dd/'QQQ_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds','QQQ.Close':'qqq_close'})
    qqq['qqq_return']=np.log(qqq['qqq_close']/qqq['qqq_close'].shift(1))
    df = pd.merge(tqqq, qqq[['ds','qqq_close','qqq_return']], on='ds')
    df['y_lag1']=df['y'].shift(1); df['y_lag2']=df['y'].shift(2)
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['stoch_k','stoch_d','qqq_close','qqq_return','y_lag1','y_lag2']
    return df[['ds','y']+regs], regs, 'level'

def prep_UPRO(dd: Path):
    upro = pd.read_csv(dd/'UPRO_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds','UPRO.Close':'y'})
    st = StochasticOscillator(close=upro['y'], high=upro['UPRO.High'], low=upro['UPRO.Low'], window=14, smooth_window=3)
    upro['stoch_k']=st.stoch(); upro['stoch_d']=st.stoch_signal()
    scaler = StandardScaler(); upro['volume_scaled']=scaler.fit_transform(upro[['UPRO.Volume']])
    spy = pd.read_csv(dd/'SPY_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds','SPY.Close':'spy_close'})
    spy['spy_return']=np.log(spy['spy_close']/spy['spy_close'].shift(1))
    df = pd.merge(upro, spy[['ds','spy_close','spy_return']], on='ds')
    df['y_lag1']=df['y'].shift(1); df['y_lag2']=df['y'].shift(2)
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['stoch_k','stoch_d','spy_close','spy_return','y_lag1','y_lag2']
    return df[['ds','y']+regs], regs, 'level'

def prep_SOXL(dd: Path):
    soxl = pd.read_csv(dd/'SOXL_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds','SOXL.Close':'y'})
    st = StochasticOscillator(close=soxl['y'], high=soxl['SOXL.High'], low=soxl['SOXL.Low'], window=14, smooth_window=3)
    soxl['stoch_k']=st.stoch(); soxl['stoch_d']=st.stoch_signal()
    scaler = StandardScaler(); soxl['volume_scaled']=scaler.fit_transform(soxl[['SOXL.Volume']])
    soxl['y_lag1']=soxl['y'].shift(1); soxl['y_lag2']=soxl['y'].shift(2)
    soxl = soxl.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['stoch_k','stoch_d','volume_scaled','y_lag1','y_lag2']
    return soxl[['ds','y']+regs], regs, 'level'

def prep_QQQ(dd: Path):
    qqq = pd.read_csv(dd/'QQQ_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds','QQQ.Close':'y'})
    qqq['qqq_return']=np.log(qqq['y']/qqq['y'].shift(1))
    qqq['y_lag1']=qqq['y'].shift(1); qqq['y_lag2']=qqq['y'].shift(2)
    st = StochasticOscillator(close=qqq['y'], high=qqq['QQQ.High'], low=qqq['QQQ.Low'])
    qqq['stoch_k']=st.stoch(); qqq['stoch_d']=st.stoch_signal()
    tqqq = pd.read_csv(dd/'TQQQ_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds'})
    tqqq['tqqq_return']=np.log(tqqq['TQQQ.Close']/tqqq['TQQQ.Close'].shift(1))
    tqqq['tqqq_lag1']=tqqq['TQQQ.Close'].shift(1)
    stt = StochasticOscillator(close=tqqq['TQQQ.Close'], high=tqqq['TQQQ.High'], low=tqqq['TQQQ.Low'])
    tqqq['tqqq_stoch_k']=stt.stoch(); tqqq['tqqq_stoch_d']=stt.stoch_signal()
    df = pd.merge(qqq, tqqq[['ds','TQQQ.Close','tqqq_return','tqqq_lag1','tqqq_stoch_k','tqqq_stoch_d']],
                  on='ds').rename(columns={'TQQQ.Close':'tqqq_close'})
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['qqq_return','y_lag1','y_lag2','stoch_k','stoch_d',
            'tqqq_close','tqqq_return','tqqq_lag1','tqqq_stoch_k','tqqq_stoch_d']
    return df[['ds','y']+regs], regs, 'level'

def prep_RHHBY(dd: Path):
    df = pd.read_csv(dd/'RHHBY_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds','RHHBY.Close':'y'}
    ).sort_values('ds').reset_index(drop=True)
    df['rhhby_return']=np.log(df['y']/df['y'].shift(1))
    df['y_lag1']=df['y'].shift(1); df['y_lag2']=df['y'].shift(2)
    xlv = pd.read_csv(dd/'XLV_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds'})
    xlv['xlv_return']=np.log(xlv['XLV.Close']/xlv['XLV.Close'].shift(1))
    xlv['xlv_lag1']=xlv['XLV.Close'].shift(1)
    df = pd.merge(df, xlv[['ds','XLV.Close','xlv_return','xlv_lag1']], on='ds').rename(columns={'XLV.Close':'xlv_close'})
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['rhhby_return','y_lag1','y_lag2','xlv_close','xlv_return','xlv_lag1']
    return df[['ds','y']+regs], regs, 'logexp'

def prep_BTC(dd: Path):
    df = pd.read_csv(dd/'BTC_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds','BTC.USD.Close':'y'}
    ).sort_values('ds').reset_index(drop=True)
    df['btc_return']=np.log(df['y']/df['y'].shift(1))
    df['y_lag1']=df['y'].shift(1); df['y_lag2']=df['y'].shift(2)
    ibit = pd.read_csv(dd/'IBIT_data.csv', parse_dates=['Date']).rename(columns={'Date':'ds'})
    ibit['ibit_return']=np.log(ibit['IBIT.Close']/ibit['IBIT.Close'].shift(1))
    ibit['ibit_lag1']=ibit['IBIT.Close'].shift(1)
    df = pd.merge(df, ibit[['ds','IBIT.Close','ibit_return','ibit_lag1']], on='ds').rename(columns={'IBIT.Close':'ibit_close'})
    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['btc_return','y_lag1','y_lag2','ibit_close','ibit_return','ibit_lag1']
    return df[['ds','y']+regs], regs, 'logexp'

def prep_UDOW(dd: Path):
    
    udow = pd.read_csv(dd/'UDOW_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds', 'UDOW.Close':'y'}
    ).sort_values('ds').reset_index(drop=True)

    # Indicadores técnicos de UDOW
    st = StochasticOscillator(
        close=udow['y'],
        high=udow['UDOW.High'],
        low=udow['UDOW.Low'],
        window=14, smooth_window=3
    )
    udow['stoch_k'] = st.stoch()
    udow['stoch_d'] = st.stoch_signal()
    scaler = StandardScaler()
    udow['volume_scaled'] = scaler.fit_transform(udow[['UDOW.Volume']])
# Regresores de DIA
    dia = pd.read_csv(dd/'DIA_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds', 'DIA.Close':'dia_close'}
    ).sort_values('ds').reset_index(drop=True)
    dia['dia_return'] = np.log(dia['dia_close'] / dia['dia_close'].shift(1))

    # Merge + rezagos
    df = pd.merge(udow, dia[['ds', 'dia_close', 'dia_return']], on='ds', how='inner')
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)

    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['stoch_k', 'stoch_d', 'volume_scaled', 'dia_close', 'dia_return', 'y_lag1', 'y_lag2']
    return df[['ds', 'y'] + regs], regs, 'level'


def prep_TSLG(dd: Path):
  
    tslg = pd.read_csv(dd/'TSLG_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds', 'TSLG.Close':'y'}
    ).sort_values('ds').reset_index(drop=True)

    # Indicadores técnicos de TSLG
    st = StochasticOscillator(
        close=tslg['y'],
        high=tslg['TSLG.High'],
        low=tslg['TSLG.Low'],
        window=14, smooth_window=3
    )
    tslg['stoch_k'] = st.stoch()
    tslg['stoch_d'] = st.stoch_signal()
    scaler = StandardScaler()
    tslg['volume_scaled'] = scaler.fit_transform(tslg[['TSLG.Volume']])

    # Regresores de TSLA
    tsla = pd.read_csv(dd/'TSLA_data.csv', parse_dates=['Date']).rename(
        columns={'Date':'ds', 'TSLA.Close':'tsla_close'}
    ).sort_values('ds').reset_index(drop=True)
    tsla['tsla_return'] = np.log(tsla['tsla_close'] / tsla['tsla_close'].shift(1))

    # Merge + rezagos
    df = pd.merge(tslg, tsla[['ds', 'tsla_close', 'tsla_return']], on='ds', how='inner')
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)

    df = df.dropna().sort_values('ds').reset_index(drop=True)
    regs = ['stoch_k', 'stoch_d', 'volume_scaled', 'tsla_close', 'tsla_return', 'y_lag1', 'y_lag2']
    return df[['ds', 'y'] + regs], regs, 'level'

# ---------- Semáforo (score con RSI/MACD/CCI + dirección Prophet) ----------
def semaforo_from_components(rsi_val, macd_val, cci_val, direccion_prophet):
    rsi_score  = 1 if 30 < rsi_val < 60 else 0
    macd_score = 1 if macd_val > 0 else 0
    cci_score  = 1 if cci_val > 0 else 0
    prop_score = 1 if direccion_prophet == 'sube' else 0 if direccion_prophet == 'baja' else 0.5
    score = W_RSI*rsi_score + W_MACD*macd_score + W_CCI*cci_score + W_PROP*prop_score
    if score > TH_LONG:
        return 'verde', round(score, 2)
    if score < TH_SHORT:
        return 'rojo', round(score, 2)
    return 'amarillo', round(score, 2)

# ---------- Backtest por activo ----------
def backtest_symbol(name: str, prep_func, n_days: int = N_DAYS):
    df, regs, mode = prep_func(DATA_DIR)
    test_days = last_n_business_days_present(df['ds'], n=n_days)
    rows = []

    for d in test_days:
        # Historial hasta el día anterior
        train = df[df['ds'] < pd.Timestamp(d)].copy()
        if len(train) < 50:
            continue

        # 1) Predicción Prophet + RF (idéntico a pipeline)
        one = prophet_rf_one_day(train, df, regs, pd.Timestamp(d), mode)
        if one is None:
            continue

        # 2) Indicadores técnicos calculados SOLO con historial disponible
        hist = df[df['ds'] <= pd.Timestamp(d)].copy()
        hist = hist.sort_values('ds').reset_index(drop=True)
        close_series = hist['y']
        # RSI, MACD y CCI sobre el histórico (último valor es el vigente en d-1 para operar en d)
        rsi = RSIIndicator(close=close_series, window=14).rsi().iloc[-1]
        macd = MACD(close=close_series).macd_diff().iloc[-1]
        # Para CCI necesitamos High/Low si existen; si no, aproximamos con y como close
        if f"{name}.High".upper() in [c.upper() for c in hist.columns]:
            # si viniera en el df (no suele estar tras prep), evitamos KeyError
            high_col = [c for c in hist.columns if c.upper()==f"{name}.High".upper()][0]
            low_col  = [c for c in hist.columns if c.upper()==f"{name}.Low".upper()][0]
            cci = CCIIndicator(high=hist[high_col], low=hist[low_col], close=close_series, window=20).cci().iloc[-1]
        else:
            # aproximación segura
            cci = CCIIndicator(high=close_series, low=close_series, close=close_series, window=20).cci().iloc[-1]

        # 3) Semáforo con score ponderado
        sem_color, sem_score = semaforo_from_components(rsi, macd, cci, one['direccion_pred'])

        rows.append({
            'asset': name, 'ds': pd.Timestamp(d),
            'prev_close': one['prev_close'], 'y_real': one['y_real'], 'y_pred': one['y_pred'],
            'yhat_lower': one['yhat_lower'], 'yhat_upper': one['yhat_upper'], 'in_ci': one['in_ci'],
            'delta_pred_rel': one['delta_pred_rel'], 'delta_real_rel': one['delta_real_rel'],
            'direccion_pred': one['direccion_pred'], 'direccion_real': one['direccion_real'],
            'rsi': float(rsi), 'macd': float(macd), 'cci': float(cci),
            'semaforo': sem_color, 'score': sem_score
        })

    bt = pd.DataFrame(rows).sort_values('ds').reset_index(drop=True)

    # Métricas y simulación (all-in por semáforo)
    msum = metrics_summary(bt) if len(bt) else metrics_summary(pd.DataFrame())
    bt, final_eq, total_ret = simulate_equity(bt, initial=1000.0, fee_bps=FEE_BPS)

    # Guardar CSV por activo
    out_csv = OUT_DIR / f"{name}_backtest_semaforo.csv"
    bt.to_csv(out_csv, index=False)

    summary = {
        'asset': name,
        'n_days': int(len(bt)),
        'final_equity': round(final_eq, 2),
        'total_return_pct': round(total_ret, 2),
        **{k: (round(v, 6) if isinstance(v, (int,float,np.floating)) else v) for k,v in msum.items()}
    }
    print(f"[{name}] -> {out_csv} | final_equity={summary['final_equity']} ({summary['total_return_pct']}%)")
    return summary

def main():
    summaries = []
    summaries.append(backtest_symbol("TQQQ", prep_TQQQ, n_days=N_DAYS))
    summaries.append(backtest_symbol("UPRO", prep_UPRO, n_days=N_DAYS))
    summaries.append(backtest_symbol("SOXL", prep_SOXL, n_days=N_DAYS))
    summaries.append(backtest_symbol("QQQ",  prep_QQQ,  n_days=N_DAYS))
    summaries.append(backtest_symbol("RHHBY", prep_RHHBY, n_days=N_DAYS))
    summaries.append(backtest_symbol("BTC",  prep_BTC,  n_days=N_DAYS))
    summaries.append(backtest_symbol("UDOW", prep_UDOW, n_days=N_DAYS))
    summaries.append(backtest_symbol("TSLG", prep_TSLG, n_days=N_DAYS))
    

    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(OUT_DIR / "summary_semaforo.csv", index=False)
    print(f"Resumen global -> {OUT_DIR / 'summary_semaforo.csv'}")

if __name__ == "__main__":
    main()
