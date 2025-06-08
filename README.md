![Preview](BBCA_BG.png)
# BCA Stock Price Forecasting using ANN

This project aims to forecast the stock price of Bank Central Asia (BCA) using Artificial Neural Networks (ANN). The prediction is based on historical stock price data retrieved from Yahoo Finance.

## 📂 Dataset

- **Source**: [Yahoo Finance](https://finance.yahoo.com/)
- **Period**: January 1, 2019 – May 1, 2025
- **Features**: Open, High, Low, Close, Volume, etc.
- The dataset was preprocessed and normalized before modeling.

## ⚙️ Tools & Libraries

- Python
- Pandas, NumPy
- scikit-learn
- TensorFlow / Keras (ANN model)
- Streamlit (for deployment)

## 🔍 Analysis Process

1. Data collection from Yahoo Finance
2. Data preprocessing and normalization using Min-Max Scaling
3. Model development using Artificial Neural Networks (ANN)
4. Model evaluation using MAE, RMSE, and R² metrics
5. Deployment using Streamlit for interactive user interface

## ✅ Results & Insights

- **Mean Absolute Error (MAE)**: 0.4344  
- **Root Mean Squared Error (RMSE)**: 0.4969  
- **R² Score**: 0.9841

The ANN model was able to achieve high accuracy in forecasting the BCA stock price. The strong R² score indicates that the model fits the data well.

## 📌 How to Run

1. Clone this repository:  
   `git clone https://github.com/jesselynmu/predict_bbca.git`
2. Install required dependencies:  
   `pip install -r requirements.txt`
3. Run the Streamlit app:  
   `streamlit run app.py`

## ✍️ Author

Jesselyn Mu  
[LinkedIn](https://www.linkedin.com/in/jesselyn-mu/) • [GitHub](https://github.com/jesselynmu)
