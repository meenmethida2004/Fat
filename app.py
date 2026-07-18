"""
app.py
======
เว็บแอปพลิเคชัน Flask สำหรับทำนาย "ระดับโรคอ้วน" (Obesity Level)
โดยโหลดโมเดล SVM ที่ฝึกและบันทึกไว้แล้ว (model/obesity_svm_model.pkl)
มาใช้ทำนายจากข้อมูลที่ผู้ใช้กรอกผ่านฟอร์มบนเว็บ

วิธีรัน (local):
    pip install -r requirements.txt
    python app.py
    เปิดเบราว์เซอร์ที่ http://127.0.0.1:5000

วิธี deploy (เช่น Render / Railway / Heroku):
    ใช้คำสั่ง start: gunicorn app:app
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# โหลดโมเดล (Pipeline เดียว: preprocessing + SVM) ตอนเริ่มแอป
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "obesity_svm_model.pkl")
model = joblib.load(MODEL_PATH)

NUMERIC_FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_FEATURES = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# คำอธิบายผลลัพธ์ภาษาไทย เพื่อแสดงผลให้ผู้ใช้เข้าใจง่าย
LABEL_TH = {
    "Insufficient_Weight": "น้ำหนักน้อยกว่าเกณฑ์",
    "Normal_Weight": "น้ำหนักปกติ",
    "Overweight_Level_I": "น้ำหนักเกิน ระดับ 1",
    "Overweight_Level_II": "น้ำหนักเกิน ระดับ 2",
    "Obesity_Type_I": "โรคอ้วน ระดับ 1",
    "Obesity_Type_II": "โรคอ้วน ระดับ 2",
    "Obesity_Type_III": "โรคอ้วน ระดับ 3 (รุนแรง)",
}


def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """แปลงข้อมูลจากฟอร์ม (dict) ให้เป็น DataFrame 1 แถว ตามลำดับ/ชนิดคอลัมน์ที่โมเดลต้องการ"""
    row = {}
    for f in NUMERIC_FEATURES:
        row[f] = [float(form_data.get(f))]
    for f in CATEGORICAL_FEATURES:
        row[f] = [form_data.get(f)]
    return pd.DataFrame(row)[FEATURES]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X_new = build_input_dataframe(request.form)
        pred_class = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        classes = model.classes_
        prob_dict = {LABEL_TH.get(c, c): round(float(p) * 100, 2)
                     for c, p in sorted(zip(classes, proba), key=lambda x: -x[1])}

        result = {
            "prediction": pred_class,
            "prediction_th": LABEL_TH.get(pred_class, pred_class),
            "probabilities": prob_dict,
        }
        return render_template("index.html", result=result, form_data=request.form)
    except Exception as e:
        return render_template("index.html", error=str(e), form_data=request.form)


# Endpoint แบบ JSON API เผื่อต้องเรียกจากระบบอื่น (เช่น mobile app / frontend แยก)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        X_new = build_input_dataframe(data)
        pred_class = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0]
        classes = model.classes_
        prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        return jsonify({
            "prediction": pred_class,
            "prediction_th": LABEL_TH.get(pred_class, pred_class),
            "probabilities": prob_dict,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
