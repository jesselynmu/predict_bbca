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

# Load scaler & model
with open("minmax_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = load_model("ann_model_bca.h5", custom_objects={'mse': MeanSquaredError()})

# Download data dari Yahoo Finance
# start_date = "2025-01-01"
# end_date = datetime.today().strftime('%Y-%m-%d')

st.subheader("Pilih Rentang Tanggal")

start_date = st.date_input("Tanggal Mulai", datetime.date(2025, 1, 1))
end_date = st.date_input("Tanggal Akhir", datetime.date.today())

if start_date > end_date:
    st.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

# Validasi agar tidak salah input
if start_date > end_date:
    st.error("Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

data = yf.download("BBCA.JK", start=start_date, end=end_date, progress=False)
data = data.reset_index()
data = data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
data.dropna(inplace=True)

# Scaling
scaled_data = scaler.transform(data[['Close', 'High', 'Low', 'Open', 'Volume']])
df_scaled = pd.DataFrame(scaled_data, columns=['close', 'high', 'low', 'open', 'volume'])
df_scaled['date'] = data['Date'].values
df_scaled['target'] = data['Close'].values  # as actual target for evaluation

# Prediksi
X_pred = df_scaled[['close', 'high', 'low', 'open', 'volume']]
y_true = df_scaled['target'].values
y_pred = model.predict(X_pred).flatten()

# Evaluasi
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# Gabungkan hasil prediksi
df_result = pd.DataFrame({
    'date': df_scaled['date'],
    'actual': y_true,
    'predicted': y_pred
})

# Group per minggu
df_result['week'] = df_result['date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_result = df_result.groupby('week')[['actual', 'predicted']].mean().reset_index()

# Tampilkan metrik
st.subheader("Evaluasi Model (Data Terbaru)")
st.write(f"**MAE**: {mae:.4f}")
st.write(f"**RMSE**: {rmse:.4f}")
st.write(f"**R² Score**: {r2:.4f}")

# Plot hasil prediksi mingguan
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
