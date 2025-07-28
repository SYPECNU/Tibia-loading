import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 用于加载模型

# 加载模型
model = joblib.load("model.pkl")

# 定义特征名称
feature_columns = [
    'Gluteus medius', 'Gluteus minimus', 'Adductor longus', 'Adductor brevis',
    'Adductor magnus', 'Tensor fasciae latae', 'Gracilis', 'Gluteus maximus',
    'Psoas major', 'Quadriceps', 'Piriformis', 'Ipsilateral erector spinae',
    'Contralateral erector spinae', 'Ipsilateral internal oblique',
    'Contralateral internal oblique', 'Ipsilateral external oblique',
    'Contralateral external oblique'
]

# 页面标题和说明
st.title("Tibial Load Prediction Based on Muscle Strength")
st.write("""
This application is designed to predict and evaluate individual tibial load risk based on personalized lower limb and trunk muscle strength.  
Please enter your muscle strength values on the left (in N/kg, range from -10 to 200).  
The model will provide an estimated tibial load to support biomechanical assessment and injury prevention.
""")

# 侧边栏输入
st.sidebar.header("Input Parameters")
input_values = []
for col in feature_columns:
    val = st.sidebar.slider(col, -10.0, 200.0, 0.0)
    input_values.append(val)

# 构造输入 DataFrame
user_input_df = pd.DataFrame([input_values], columns=feature_columns)

# 模型预测
predicted_load = model.predict(user_input_df)

# 显示预测结果
st.subheader("Prediction Result")
st.write(f"**Predicted Tibial Load:** `{predicted_load[0]:.2f}` N/kg")
