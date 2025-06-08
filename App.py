import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime

st.title("Prediksi Harga Saham BBCA (Per Minggu)")

with open("minmax_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = load_model("ann_model_bca.h5", custom_objects={'mse': MeanSquaredError()})

st.subheader("Pilih Rentang Tanggal")

start_date = st.date_input("Tanggal Mulai", datetime.date(2025, 1, 1))
end_date = st.date_input("Tanggal Akhir", datetime.date.today())

if start_date > end_date:
    st.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

data = yf.download("BBCA.JK", start=start_date, end=end_date, progress=False)
data = data.reset_index()
data = data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

scaled_data = scaler.transform(data[['Close', 'High', 'Low', 'Open', 'Volume']])
df_scaled = pd.DataFrame(scaled_data, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
df_scaled['Date'] = data['Date'].values
df_scaled['Target'] = data['Close'].values  

X_pred = df_scaled[['High', 'Low', 'Open', 'Volume']]
y_true = df_scaled['Target'].values
y_pred = model.predict(X_pred).flatten()

# Evaluasi
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

df_result = pd.DataFrame({
    'Date': df_scaled['Date'],
    'actual': y_true,
    'predicted': y_pred
})

df_result['week'] = df_result['Date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_result = df_result.groupby('week')[['actual', 'predicted']].mean().reset_index()

st.subheader("Evaluasi Model (Data Terbaru)")
st.write(f"**MAE**: {mae:.4f}")
st.write(f"**RMSE**: {rmse:.4f}")
st.write(f"**R² Score**: {r2:.4f}")

st.subheader("Grafik Prediksi vs Aktual (Per Minggu)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekly_result['week'], weekly_result['actual'], label='Actual', marker='o')
ax.plot(weekly_result['week'], weekly_result['predicted'], label='Predicted', marker='x')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
plt.xlabel("Minggu")
plt.ylabel("Harga (Close)")
plt.title("Prediksi Mingguan BBCA")
plt.legend()
plt.tight_layout()
st.pyplot(fig)