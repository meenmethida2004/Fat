"""
app.py
======
เว็บแอป Streamlit สำหรับทำนาย "ระดับโรคอ้วน" (Obesity Level)
โดยโหลดโมเดล SVM ที่ฝึกและบันทึกไว้แล้ว (obesity_svm_model.pkl)
มาใช้ทำนายจากข้อมูลที่ผู้ใช้กรอกผ่านฟอร์มบนเว็บ

วิธีรัน (local):
    pip install -r requirements.txt
    streamlit run app.py

วิธี deploy บน Streamlit Cloud:
    1. push โค้ดทั้งหมด (app.py, obesity_svm_model.pkl, requirements.txt) ขึ้น GitHub repo
       (ไฟล์ .pkl ต้องอยู่ "โฟลเดอร์เดียวกัน" กับ app.py ตามที่กำหนดไว้ด้านล่าง)
    2. ไปที่ https://share.streamlit.io -> New app -> เลือก repo -> Main file: app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ทำนายระดับโรคอ้วน (SVM)", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------------------
# โหลดโมเดล (Pipeline เดียว: preprocessing + SVM) — cache ไว้ไม่ต้องโหลดซ้ำทุกครั้ง
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "obesity_svm_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

NUMERIC_FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_FEATURES = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL_TH = {
    "Insufficient_Weight": "น้ำหนักน้อยกว่าเกณฑ์",
    "Normal_Weight": "น้ำหนักปกติ",
    "Overweight_Level_I": "น้ำหนักเกิน ระดับ 1",
    "Overweight_Level_II": "น้ำหนักเกิน ระดับ 2",
    "Obesity_Type_I": "โรคอ้วน ระดับ 1",
    "Obesity_Type_II": "โรคอ้วน ระดับ 2",
    "Obesity_Type_III": "โรคอ้วน ระดับ 3 (รุนแรง)",
}

# ---------------------------------------------------------------------------
# หน้าเว็บ
# ---------------------------------------------------------------------------
st.title("🩺 ระบบทำนายระดับโรคอ้วนด้วย SVM")
st.caption("กรอกข้อมูลสุขภาพและพฤติกรรมด้านล่าง เพื่อทำนายระดับความเสี่ยงโรคอ้วน")

with st.form("predict_form"):
    st.subheader("ข้อมูลทั่วไป")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
        age = st.number_input("อายุ (Age)", min_value=10, max_value=100, value=25)
    with col2:
        height = st.number_input("ส่วนสูง (Height) เมตร", min_value=1.2, max_value=2.3, value=1.70, step=0.01)
        weight = st.number_input("น้ำหนัก (Weight) กก.", min_value=20.0, max_value=250.0, value=65.0, step=0.1)

    st.subheader("พฤติกรรมการกิน")
    col3, col4 = st.columns(2)
    with col3:
        family_history = st.selectbox("ประวัติครอบครัวเป็นโรคอ้วน", ["yes", "no"])
        favc = st.selectbox("กินอาหารแคลอรีสูงบ่อย (FAVC)", ["yes", "no"])
        fcvc = st.slider("ความถี่กินผัก (FCVC)", 1.0, 3.0, 2.0, step=0.1)
    with col4:
        ncp = st.slider("จำนวนมื้ออาหารหลัก/วัน (NCP)", 1.0, 4.0, 3.0, step=0.1)
        caec = st.selectbox("กินจุบจิบระหว่างมื้อ (CAEC)", ["no", "Sometimes", "Frequently", "Always"], index=1)
        scc = st.selectbox("นับแคลอรีอาหาร (SCC)", ["no", "yes"])

    st.subheader("พฤติกรรมสุขภาพอื่น ๆ")
    col5, col6 = st.columns(2)
    with col5:
        smoke = st.selectbox("สูบบุหรี่ (SMOKE)", ["no", "yes"])
        ch2o = st.slider("ปริมาณน้ำดื่ม/วัน (CH2O) ลิตร", 1.0, 3.0, 2.0, step=0.1)
        faf = st.slider("ความถี่ออกกำลังกาย (FAF)", 0.0, 3.0, 1.0, step=0.1)
    with col6:
        tue = st.slider("เวลาใช้เทคโนโลยี/วัน (TUE) ชม.", 0.0, 3.0, 1.0, step=0.1)
        calc = st.selectbox("ดื่มแอลกอฮอล์ (CALC)", ["no", "Sometimes", "Frequently"], index=1)
        mtrans = st.selectbox(
            "การเดินทางหลัก (MTRANS)",
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        )

    submitted = st.form_submit_button("ทำนายผล", use_container_width=True)

if submitted:
    input_dict = {
        "Age": [age], "Height": [height], "Weight": [weight],
        "FCVC": [fcvc], "NCP": [ncp], "CH2O": [ch2o], "FAF": [faf], "TUE": [tue],
        "Gender": [gender], "family_history_with_overweight": [family_history],
        "FAVC": [favc], "CAEC": [caec], "SMOKE": [smoke], "SCC": [scc],
        "CALC": [calc], "MTRANS": [mtrans],
    }
    X_new = pd.DataFrame(input_dict)[FEATURES]

    try:
        pred_class = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        classes = model.classes_

        st.success(f"### ผลการทำนาย: {LABEL_TH.get(pred_class, pred_class)}")
        st.caption(f"รหัสคลาส: {pred_class}")

        st.subheader("ความน่าจะเป็นของแต่ละระดับ")
        prob_df = pd.DataFrame({
            "ระดับ": [LABEL_TH.get(c, c) for c in classes],
            "ความน่าจะเป็น (%)": [round(float(p) * 100, 2) for p in proba],
        }).sort_values("ความน่าจะเป็น (%)", ascending=False).reset_index(drop=True)

        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("ระดับ"))

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนาย: {e}")
