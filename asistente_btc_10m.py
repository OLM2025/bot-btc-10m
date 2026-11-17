from datetime import datetime
import os
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

SIMBOLO = "BTCUSDT"
INTERVALO = "5m"
TP_PCT = 0.0035  # +0.35% ganancia objetivo (asegura +0.30% limpio)
SL_PCT = 0.0018  # -0.18% pérdida máxima permitida
MAX_VELAS = 2  # 2 velas de 5m = 10 minutos exactos
UMBRAL_PROB = 0.37  # Con 34% empatas; con 37% tienes ventaja y operará seguido
META_PRUEBAS = 100
ARCHIVO_CSV = "registro_pruebas_btc.csv"

FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "volatilidad_rel",
    "volumen_rel",
    "dist_sma20",
    "rango_cuerpo",
    "hora",
]


def obtener_velas_binance(limite=1000):
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": SIMBOLO, "interval": INTERVALO, "limit": limite}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    cols = [
        "open_time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_time",
        "qav",
        "trades",
        "tbbav",
        "tbqav",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df["fecha"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("fecha", inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].iloc[:-1]


def crear_features(df):
    data = df.copy()
    c, h, l, o, v = (
        data["Close"],
        data["High"],
        data["Low"],
        data["Open"],
        data["Volume"],
    )

    data["ret_1"] = c.pct_change(1)
    data["ret_3"] = c.pct_change(3)
    data["ret_6"] = c.pct_change(6)
    rango = (h - l) / c
    data["volatilidad_rel"] = rango / rango.rolling(20).mean()
    data["volumen_rel"] = v / v.rolling(20).mean()
    sma_20 = c.rolling(20).mean()
    data["dist_sma20"] = (c - sma_20) / sma_20
    data["rango_cuerpo"] = (c - o) / (h - l + 1e-9)
    data["hora"] = data.index.hour
    return data


def etiquetar_triple_barrera(df):
    closes, highs, lows = (
        df["Close"].values,
        df["High"].values,
        df["Low"].values,
    )
    n = len(df)
    target = np.zeros(n, dtype=int)

    for i in range(n - MAX_VELAS):
        entrada = closes[i]
        tp = entrada * (1 + TP_PCT)
        sl = entrada * (1 - SL_PCT)
        for j in range(1, MAX_VELAS + 1):
            if lows[i + j] <= sl:
                target[i] = 0
                break
            if highs[i + j] >= tp:
                target[i] = 1
                break

    df["target"] = target
    return df.iloc[:-MAX_VELAS]


def entrenar_modelo(df_hist):
    df_feat = crear_features(df_hist)
    df_model = etiquetar_triple_barrera(df_feat).dropna()
    X, y = df_model[FEATURES], df_model["target"]

    base = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=150, random_state=42
    )
    modelo = CalibratedClassifierCV(base, method="isotonic", cv=3)
    modelo.fit(X, y)
    return modelo


def cargar_registro():
    if os.path.exists(ARCHIVO_CSV):
        return pd.read_csv(ARCHIVO_CSV)
    return pd.DataFrame(
        columns=[
            "fecha_vela",
            "par",
            "entrada",
            "probabilidad_pct",
            "tp_objetivo",
            "sl_limite",
            "resultado",
        ]
    )


def auditar_pendientes(reg, df_actual):
    if reg.empty:
        return reg

    for idx, row in reg[reg["resultado"] == "PENDIENTE"].iterrows():
        t_entrada = pd.to_datetime(row["fecha_vela"])
        siguientes = df_actual[df_actual.index > t_entrada].head(MAX_VELAS)
        if siguientes.empty:
            continue

        for _, vela in siguientes.iterrows():
            if vela["Low"] <= row["sl_limite"]:
                reg.at[idx, "resultado"] = "PERDIDA"
                break
            if vela["High"] >= row["tp_objetivo"]:
                reg.at[idx, "resultado"] = "GANADA"
                break

        if (
            reg.at[idx, "resultado"] == "PENDIENTE"
            and len(siguientes) >= MAX_VELAS
        ):
            cierre_final = siguientes.iloc[-1]["Close"]
            retorno_r = (cierre_final - row["entrada"]) / (
                row["entrada"] * SL_PCT
            )
            reg.at[idx, "resultado"] = f"EXPIRADA ({retorno_r:+.2f})"

    return reg


def ejecutar():
    reg = cargar_registro()
    cerradas = reg[reg["resultado"] != "PENDIENTE"]

    if len(cerradas) >= META_PRUEBAS:
        print(
            f"META COMPLETADA: Ya se registraron {len(cerradas)} pruebas. Revisa"
            f" {ARCHIVO_CSV}."
        )
        return

    df_hist = obtener_velas_binance(limite=1000)
    reg = auditar_pendientes(reg, df_hist.tail(100))

    modelo = entrenar_modelo(df_hist)
    df_feat = crear_features(df_hist.tail(100)).dropna()

    ultima_vela = df_feat.iloc[[-1]]
    fecha_str = str(ultima_vela.index[0])
    precio = float(ultima_vela["Close"].iloc[0])
    prob = float(modelo.predict_proba(ultima_vela[FEATURES])[0, 1])

    ya_registrada = (
        (reg["fecha_vela"] == fecha_str).any() if not reg.empty else False
    )

    if prob >= UMBRAL_PROB and not ya_registrada:
        tp = round(precio * (1 + TP_PCT), 2)
        sl = round(precio * (1 - SL_PCT), 2)
        nueva_fila = pd.DataFrame([{
            "fecha_vela": fecha_str,
            "par": SIMBOLO,
            "entrada": round(precio, 2),
            "probabilidad_pct": round(prob * 100, 1),
            "tp_objetivo": tp,
            "sl_limite": sl,
            "resultado": "PENDIENTE",
        }])
        reg = pd.concat([reg, nueva_fila], ignore_index=True)
        decision = (
            f"COMPRAR AHORA (Vender con ganancia en ${tp:,.2f} | Cortar pérdida"
            f" en ${sl:,.2f})"
        )
    elif ya_registrada:
        decision = "YA EVALUADA (Esta vela de 5 min ya fue registrada)"
    else:
        decision = "ESPERAR (Probabilidad muy baja para arriesgar dinero)"

    reg.to_csv(ARCHIVO_CSV, index=False)

    cerradas = reg[reg["resultado"] != "PENDIENTE"]
    ganadas = (cerradas["resultado"] == "GANADA").sum()
    perdidas = (cerradas["resultado"] == "PERDIDA").sum()
    puntos = (ganadas * 1.94) - (perdidas * 1.0)

    print("=== REPORTE DEL ASISTENTE (CADA 5 MIN) ===")
    print(f"Precio actual Bitcoin: ${precio:,.2f}")
    print(f"Probabilidad de ganar: {prob*100:.1f}% (Mínimo exigido: {UMBRAL_PROB*100:.1f}%)")
    print(f"DECISIÓN: {decision}")
    print(
        f"PROGRESO: {len(cerradas)} de {META_PRUEBAS} pruebas terminadas |"
        f" Ganadas: {ganadas} | Perdidas: {perdidas} | Puntos netos:"
        f" {puntos:+.2f}"
    )


if __name__ == "__main__":
    ejecutar()
