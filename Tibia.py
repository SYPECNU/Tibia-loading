import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap

# 载入数据
@st.cache_data
def load_data():
    data1 = pd.read_csv("MTSS-110.csv", encoding='gbk')  # 请确保加速.csv已经上传
    data1.dropna(inplace=True)
    data1.columns = [
        'Number',
        'Target variable',
        'Gluteus medius',
        'Gluteus minimus',
        'Adductor longus',
        'Adductor brevis',
        'Adductor magnus',
        'Tensor fasciae latae',
        'Gracilis',
        'Gluteus maximus',
        'Psoas major',
        'Quadriceps',
        'Piriformis',
        'Ipsilateral erector spinae',
        'Contralateral erector spinae',
        'Ipsilateral internal oblique',
        'Contralateral internal oblique',
        'Ipsilateral external oblique',
        'Contralateral external oblique'
    ]
    return data1

data1 = load_data()

# 提取特征和标签
X = data1[[
    'Gluteus medius',
    'Gluteus minimus',
    'Adductor longus',
    'Adductor brevis',
    'Adductor magnus',
    'Tensor fasciae latae',
    'Gracilis',
    'Gluteus maximus',
    'Psoas major',
    'Quadriceps',
    'Piriformis',
    'Ipsilateral erector spinae',
    'Contralateral erector spinae',
    'Ipsilateral internal oblique',
    'Contralateral internal oblique',
    'Ipsilateral external oblique',
    'Contralateral external oblique'
]]

y = data1[['Target variable']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=1,
    min_child_weight=3,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.7,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# 创建Streamlit界面
st.title("Tibial Load Prediction Based on Muscle Strength")

# 显示应用说明
st.write("""
This application is designed to predict and evaluate individual tibial load risk based on personalized lower limb and trunk muscle strength.  
Please enter your muscle strength values on the left (in N/kg, range from -10 to 200).  
The model will provide an estimated tibial load to support biomechanical assessment and injury prevention.
""")

# 在侧边栏中添加用户输入
st.sidebar.header("Input Parameters")

# 各种特征的输入控件
gluteus_medius = st.sidebar.slider("Gluteus medius", min_value=-10.0, max_value=200.0, value=0.0)
gluteus_minimus = st.sidebar.slider("Gluteus minimus", min_value=-10.0, max_value=200.0, value=0.0)
adductor_longus = st.sidebar.slider("Adductor longus", min_value=-10.0, max_value=200.0, value=0.0)
adductor_brevis = st.sidebar.slider("Adductor brevis", min_value=-10.0, max_value=200.0, value=0.0)
adductor_magnus = st.sidebar.slider("Adductor magnus", min_value=-10.0, max_value=200.0, value=0.0)
tensor_fasciae_latae = st.sidebar.slider("Tensor fasciae latae", min_value=-10.0, max_value=200.0, value=0.0)
gracilis = st.sidebar.slider("Gracilis", min_value=-10.0, max_value=200.0, value=0.0)
gluteus_maximus = st.sidebar.slider("Gluteus maximus", min_value=-10.0, max_value=200.0, value=0.0)
psoas_major = st.sidebar.slider("Psoas major", min_value=-10.0, max_value=200.0, value=0.0)
quadriceps = st.sidebar.slider("Quadriceps", min_value=-10.0, max_value=200.0, value=0.0)
piriformis = st.sidebar.slider("Piriformis", min_value=-10.0, max_value=200.0, value=0.0)
ipsi_erector_spinae = st.sidebar.slider("Ipsilateral erector spinae", min_value=-10.0, max_value=200.0, value=0.0)
contra_erector_spinae = st.sidebar.slider("Contralateral erector spinae", min_value=-10.0, max_value=200.0, value=0.0)
ipsi_internal_oblique = st.sidebar.slider("Ipsilateral internal oblique", min_value=-10.0, max_value=200.0, value=0.0)
contra_internal_oblique = st.sidebar.slider("Contralateral internal oblique", min_value=-10.0, max_value=200.0, value=0.0)
ipsi_external_oblique = st.sidebar.slider("Ipsilateral external oblique", min_value=-10.0, max_value=200.0, value=0.0)
contra_external_oblique = st.sidebar.slider("Contralateral external oblique", min_value=-10.0, max_value=200.0, value=0.0)

# 用户输入的特征值
user_input = np.array([[gluteus_medius, gluteus_minimus, adductor_longus, adductor_brevis,
                        adductor_magnus, tensor_fasciae_latae, gracilis, gluteus_maximus,
                        psoas_major, quadriceps, piriformis, ipsi_erector_spinae,
                        contra_erector_spinae, ipsi_internal_oblique, contra_internal_oblique,
                        ipsi_external_oblique, contra_external_oblique]])

# 预测
predicted_load = model.predict(user_input)

# 显示预测结果
st.write(f"Predicted Tibial Load: {predicted_load[0]:.2f} N/kg")
