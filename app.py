import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load mô hình và scaler
model = tf.keras.models.load_model("heart_disease_model.keras")
scaler = joblib.load("scaler.pkl")

# Tạo giao diện Streamlit
st.title("Dự đoán Bệnh Tim")

# Nhập dữ liệu đầu vào
age = st.number_input("Tuổi", min_value=1, max_value=120, value=30)
sex = st.selectbox("Giới tính", ["Male", "Female"])
cp = st.selectbox("Loại đau ngực", [0, 1, 2, 3])
trestbps = st.number_input("Huyết áp khi nghỉ (mmHg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dL?", [0, 1])
restecg = st.selectbox("Kết quả điện tim", [0, 1, 2])
thalch = st.number_input("Nhịp tim tối đa đạt được", min_value=50, max_value=250, value=150)
exang = st.selectbox("Đau thắt ngực khi gắng sức?", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Độ dốc của đoạn ST", [0, 1, 2])
ca = st.number_input("Số mạch máu bị hẹp (0-3)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thal (0: normal, 1: fixed defect, 2: reversable defect)", [0, 1, 2])

# Chuyển đổi dữ liệu đầu vào
sex = 1 if sex == "Male" else 0

# Tạo DataFrame cho dữ liệu đầu vào
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Chuẩn hóa dữ liệu
input_data[input_data.columns] = scaler.transform(input_data)

# Dự đoán
if st.button("Dự đoán"):
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)[0]
    probability = predictions[0][predicted_class]  

    heart_disease_levels = {
        0: "Không bị bệnh tim",
        1: "Bệnh tim mức độ nhẹ",
        2: "Bệnh tim mức độ trung bình",
        3: "Bệnh tim mức độ nặng",
        4: "Bệnh tim mức độ rất nặng"
    }

    st.write(f"**Kết quả dự đoán:** {heart_disease_levels[predicted_class]}")
    st.write(f"**Xác suất:** {probability:.2%}")
