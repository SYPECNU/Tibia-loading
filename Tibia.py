import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv("MTSS-110.csv", encoding="gbk")
    df.dropna(inplace=True)
    df.columns = [
        'Number', 'Target variable', 'Gluteus medius', 'Gluteus minimus', 'Adductor longus', 'Adductor brevis',
        'Adductor magnus', 'Tensor fasciae latae', 'Gracilis', 'Gluteus maximus',
        'Psoas major', 'Quadriceps', 'Piriformis', 'Ipsilateral erector spinae',
        'Contralateral erector spinae', 'Ipsilateral internal oblique',
        'Contralateral internal oblique', 'Ipsilateral external oblique',
        'Contralateral external oblique'
    ]
    return df

data = load_data()

# 特征和标签
feature_columns = data.columns[2:]
X = data[feature_columns]
y = data["Target variable"]

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    max_depth=3,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Streamlit 前端
st.title("Tibial Load Prediction Based on Muscle Strength")
st.write("""
Please enter your muscle strength values below (N/kg).
""")

st.sidebar.header("Input Parameters")
input_values = []
for col in feature_columns:
    val = st.sidebar.slider(col, -10.0, 200.0, 0.0)
    input_values.append(val)

user_input_df = pd.DataFrame([input_values], columns=feature_columns)

# 预测
predicted_load = model.predict(user_input_df)

# 显示结果
st.subheader("Prediction Result")
st.write(f"**Predicted Tibial Load:** `{predicted_load[0]:.2f}` N/kg")
