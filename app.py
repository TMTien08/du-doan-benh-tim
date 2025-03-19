import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image

# Load mô hình và scaler
model = tf.keras.models.load_model("heart_disease_model.keras")
scaler = joblib.load("scaler.pkl")

# Thiết lập giao diện
st.set_page_config(page_title="Dự đoán Bệnh Tim", page_icon="❤️", layout="centered")

# Tiêu đề và hình ảnh
title_col, img_col = st.columns([3, 1])
with title_col:
    st.markdown("<h1 style='text-align: center; color: red;'>Dự đoán Bệnh Tim</h1>", unsafe_allow_html=True)
with img_col:
    st.image(Image.open("heart_icon.png"), width=80)

st.markdown("---")

# Nhập dữ liệu đầu vào
st.subheader("Nhập thông tin sức khỏe của bạn")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=30)
    sex = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
    cp = st.selectbox("Loại đau ngực", ["Không đau", "Đau nhẹ", "Đau trung bình", "Đau nặng"], index=0)
    trestbps = st.number_input("Huyết áp khi nghỉ (mmHg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.radio("Đường huyết lúc đói > 120 mg/dL?", ["Không", "Có"], horizontal=True)

with col2:
    restecg = st.selectbox("Kết quả điện tim", ["Bình thường", "Bất thường nhẹ", "Bất thường nghiêm trọng"], index=0)
    thalch = st.number_input("Nhịp tim tối đa đạt được", min_value=50, max_value=250, value=150)
    exang = st.radio("Đau thắt ngực khi gắng sức?", ["Không", "Có"], horizontal=True)
    oldpeak = st.slider("ST depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Độ dốc của đoạn ST", ["Dốc xuống", "Bằng phẳng", "Dốc lên"], index=1)
    ca = st.slider("Số mạch máu bị hẹp", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thal", ["Bình thường", "Khiếm khuyết cố định", "Khiếm khuyết có thể phục hồi"], index=0)

# Chuyển đổi dữ liệu đầu vào
sex = 1 if sex == "Nam" else 0
cp = ["Không đau", "Đau nhẹ", "Đau trung bình", "Đau nặng"].index(cp)
fbs = 1 if fbs == "Có" else 0
restecg = ["Bình thường", "Bất thường nhẹ", "Bất thường nghiêm trọng"].index(restecg)
exang = 1 if exang == "Có" else 0
slope = ["Dốc xuống", "Bằng phẳng", "Dốc lên"].index(slope)
thal = ["Bình thường", "Khiếm khuyết cố định", "Khiếm khuyết có thể phục hồi"].index(thal)

# Tạo DataFrame cho dữ liệu đầu vào
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Chuẩn hóa dữ liệu
input_data[input_data.columns] = scaler.transform(input_data)

# Nút dự đoán
st.markdown("---")
if st.button("Dự đoán", use_container_width=True):
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

    st.markdown(f"""
        <div style="text-align: center; border: 2px solid red; padding: 10px; border-radius: 10px; background-color: #ffe6e6;">
            <h3 style="color: red;">{heart_disease_levels[predicted_class]}</h3>
            <h4>Xác suất: {probability:.2%}</h4>
        </div>
    """, unsafe_allow_html=True)
