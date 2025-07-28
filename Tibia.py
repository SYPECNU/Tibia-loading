import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 用于加载模型

# 加载模型
model = joblib.load("model.pkl")

# 创建Streamlit界面
st.title("Tibial Load Prediction Based on Muscle Strength")

st.write("""
This application is designed to predict and evaluate individual tibial load risk based on personalized lower limb and trunk muscle strength.  
Please enter your muscle strength values on the left (in N/kg, range from -10 to 200).  
The model will provide an estimated tibial load to support biomechanical assessment and injury prevention.
""")

# 在侧边栏中添加用户输入
st.sidebar.header("Input Parameters")

gluteus_medius = st.sidebar.slider("Gluteus medius", -10.0, 200.0, 0.0); gluteus_minimus = st.sidebar.slider("Gluteus minimus", -10.0, 200.0, 0.0)
adductor_longus = st.sidebar.slider("Adductor longus", -10.0, 200.0, 0.0); adductor_brevis = st.sidebar.slider("Adductor brevis", -10.0, 200.0, 0.0)
adductor_magnus = st.sidebar.slider("Adductor magnus", -10.0, 200.0, 0.0); tensor_fasciae_latae = st.sidebar.slider("Tensor fasciae latae", -10.0, 200.0, 0.0)
gracilis = st.sidebar.slider("Gracilis", -10.0, 200.0, 0.0); gluteus_maximus = st.sidebar.slider("Gluteus maximus", -10.0, 200.0, 0.0)
psoas_major = st.sidebar.slider("Psoas major", -10.0, 200.0, 0.0); quadriceps = st.sidebar.slider("Quadriceps", -10.0, 200.0, 0.0)
piriformis = st.sidebar.slider("Piriformis", -10.0, 200.0, 0.0); ipsi_erector_spinae = st.sidebar.slider("Ipsilateral erector spinae", -10.0, 200.0, 0.0)
contra_erector_spinae = st.sidebar.slider("Contralateral erector spinae", -10.0, 200.0, 0.0); ipsi_internal_oblique = st.sidebar.slider("Ipsilateral internal oblique", -10.0, 200.0, 0.0)
contra_internal_oblique = st.sidebar.slider("Contralateral internal oblique", -10.0, 200.0, 0.0); ipsi_external_oblique = st.sidebar.slider("Ipsilateral external oblique", -10.0, 200.0, 0.0)
contra_external_oblique = st.sidebar.slider("Contralateral external oblique", -10.0, 200.0, 0.0)

# 构造输入数据
user_input = np.array([[gluteus_medius, gluteus_minimus, adductor_longus, adductor_brevis,
                        adductor_magnus, tensor_fasciae_latae, gracilis, gluteus_maximus,
                        psoas_major, quadriceps, piriformis, ipsi_erector_spinae,
                        contra_erector_spinae, ipsi_internal_oblique, contra_internal_oblique,
                        ipsi_external_oblique, contra_external_oblique]])

# 预测
predicted_load = model.predict(user_input)

# 显示结果
st.write(f"Predicted Tibial Load: {predicted_load[0]:.2f} N/kg")
