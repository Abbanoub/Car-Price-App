import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. تحميل الموديل والـ Scaler
model = joblib.load('car_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title(" Car Price Prediction App 🚗")

# استخدمنا st.form لتجميع المدخلات ومنع التحديث العشوائي
with st.form("prediction_form"):
    st.subheader("Enter Car Details")
    
    # المدخلات بنفس أسماء الأعمدة في مشروعك
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2018)
    mileage = st.number_input("Mileage (km)", min_value=0, value=30000)
    engine_size = st.number_input("Engine Size (L)", min_value=0.1, max_value=10.0, value=1.6, step=0.1)
    
    # زر الإرسال داخل الفورم
    submit_button = st.form_submit_button("Predict Price")

# منطق التوقع
if submit_button:
    try:
        # الترتيب الصحيح والضروري جداً بناءً على ملف الـ Notebook الخاص بك:
        # 1. Year of manufacture
        # 2. Mileage
        # 3. Engine size
        features = np.array([[year, mileage, engine_size]])
        
        # تحويل البيانات باستخدام الـ Scaler
        features_scaled = scaler.transform(features)
        
        # حساب التوقع
        prediction = model.predict(features_scaled)
        
        # عرض النتيجة بشكل واضح
        st.markdown("---")
        st.success(f"### The Estimated Price is: ${prediction[0]:,.2f}")
        
        # لإثبات أن النتيجة تتغير، سنعرض القيم التي تم استخدامها
        st.info(f"Last prediction made for: Year {year}, {mileage} km, Engine {engine_size}L")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Note: Every time you change inputs, you must click 'Predict Price' to update the result.")