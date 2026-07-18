"""
train_model.py
================
โมเดล Machine Learning สำหรับทำนาย "ระดับโรคอ้วน" (Obesity Level) ด้วยอัลกอริทึม
Support Vector Machine (SVM) เขียนเป็นขั้นตอน (step) ตามกระบวนการ ML มาตรฐาน:

    STEP 1: เตรียม/โหลดชุดข้อมูล (Data Collection)
    STEP 2: สำรวจข้อมูลเบื้องต้น (EDA)
    STEP 3: เตรียมข้อมูล (Data Preprocessing)
    STEP 4: แบ่งข้อมูล Train / Test (Data Splitting)
    STEP 5: สร้าง Pipeline (Preprocessing + Model)
    STEP 6: ปรับจูนพารามิเตอร์ด้วย GridSearchCV (Hyperparameter Tuning)
    STEP 7: ประเมินผลโมเดล (Model Evaluation)
    STEP 8: บันทึกโมเดล (Model Persistence) -> ได้ไฟล์เดียว "obesity_svm_model.pkl"

หมายเหตุ: เนื่องจากสภาพแวดล้อมนี้ไม่สามารถเชื่อมต่ออินเทอร์เน็ตเพื่อดาวน์โหลดชุดข้อมูลจริงได้
สคริปต์นี้จึงมีฟังก์ชัน generate_synthetic_dataset() สำหรับ "จำลอง" ชุดข้อมูลที่มีโครงสร้าง
และความสัมพันธ์ของตัวแปรใกล้เคียงกับชุดข้อมูล Obesity Levels ที่ใช้กันจริงในงานวิจัย
(อ้างอิงแนวคิดจาก UCI "Estimation of obesity levels based on eating habits and
physical condition" dataset) ถ้าคุณมีไฟล์ CSV ข้อมูลจริงของตัวเอง สามารถแทนที่ฟังก์ชันนี้
ด้วย pd.read_csv("your_data.csv") ได้ทันที โดยคอลัมน์ต้องตรงกับ FEATURE ที่กำหนดไว้ด้านล่าง
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# คอลัมน์ตัวเลข (numeric features)
NUMERIC_FEATURES = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
# คอลัมน์ประเภท (categorical features)
CATEGORICAL_FEATURES = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
TARGET = "NObeyesdad"


# ---------------------------------------------------------------------------
# STEP 1: เตรียม/โหลดชุดข้อมูล (Data Collection)
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """จำลองชุดข้อมูลพฤติกรรมสุขภาพและระดับโรคอ้วน (ใช้แทนข้อมูลจริงในสภาพแวดล้อมนี้)"""
    genders = np.random.choice(["Male", "Female"], n_samples)
    age = np.random.randint(14, 65, n_samples)
    height = np.round(np.random.normal(1.65, 0.09, n_samples), 2)
    height = np.clip(height, 1.45, 2.05)

    family_history = np.random.choice(["yes", "no"], n_samples, p=[0.55, 0.45])
    favc = np.random.choice(["yes", "no"], n_samples, p=[0.6, 0.4])          # กินอาหารแคลอรีสูงบ่อย
    fcvc = np.round(np.random.uniform(1, 3, n_samples), 1)                    # ความถี่กินผัก
    ncp = np.round(np.random.uniform(1, 4, n_samples), 1)                     # จำนวนมื้ออาหารหลัก
    caec = np.random.choice(["no", "Sometimes", "Frequently", "Always"], n_samples,
                             p=[0.1, 0.55, 0.25, 0.10])                       # กินจุบจิบระหว่างมื้อ
    smoke = np.random.choice(["yes", "no"], n_samples, p=[0.08, 0.92])
    ch2o = np.round(np.random.uniform(1, 3, n_samples), 1)                    # ปริมาณน้ำดื่ม (ลิตร)
    scc = np.random.choice(["yes", "no"], n_samples, p=[0.2, 0.8])            # นับแคลอรีอาหาร
    faf = np.round(np.random.uniform(0, 3, n_samples), 1)                     # ความถี่ออกกำลังกาย
    tue = np.round(np.random.uniform(0, 3, n_samples), 1)                     # เวลาใช้เทคโนโลยี
    calc = np.random.choice(["no", "Sometimes", "Frequently"], n_samples,
                             p=[0.3, 0.55, 0.15])                             # ดื่มแอลกอฮอล์
    mtrans = np.random.choice(
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        n_samples, p=[0.45, 0.2, 0.25, 0.05, 0.05],
    )

    # --- สร้าง "risk score" จากพฤติกรรม เพื่อให้ BMI สัมพันธ์กับพฤติกรรมจริง ---
    risk = (
        (favc == "yes").astype(float) * 2.5
        + (family_history == "yes").astype(float) * 2.0
        + (caec == "Frequently").astype(float) * 1.5
        + (caec == "Always").astype(float) * 2.5
        + (2.5 - fcvc) * 1.2          # กินผักน้อย -> เสี่ยงมากขึ้น
        - faf * 2.0                    # ออกกำลังกายเยอะ -> เสี่ยงลดลง
        + tue * 0.8                    # ใช้เทคโนโลยีเยอะ (นั่งนาน) -> เสี่ยงเพิ่ม
        + (scc == "no").astype(float) * 1.0
        + np.random.normal(0, 2.5, n_samples)   # สุ่มความแปรปรวนตามธรรมชาติ
    )

    base_bmi = 22 + risk
    bmi = np.clip(base_bmi, 14, 55)
    weight = np.round(bmi * (height ** 2), 1)

    def bmi_to_class(b):
        if b < 18.5:
            return "Insufficient_Weight"
        elif b < 25:
            return "Normal_Weight"
        elif b < 27.5:
            return "Overweight_Level_I"
        elif b < 30:
            return "Overweight_Level_II"
        elif b < 35:
            return "Obesity_Type_I"
        elif b < 40:
            return "Obesity_Type_II"
        else:
            return "Obesity_Type_III"

    label = [bmi_to_class(b) for b in bmi]

    df = pd.DataFrame({
        "Gender": genders,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
        TARGET: label,
    })
    return df


def main():
    print("=" * 60)
    print("STEP 1: เตรียม/โหลดชุดข้อมูล")
    print("=" * 60)
    df = generate_synthetic_dataset(n_samples=2000)
    df.to_csv("obesity_dataset.csv", index=False)
    print(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    print("บันทึกชุดข้อมูลตัวอย่างไว้ที่ obesity_dataset.csv แล้ว\n")

    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: สำรวจข้อมูลเบื้องต้น (EDA)")
    print("=" * 60)
    print(df.head(), "\n")
    print("จำนวนค่าว่างในแต่ละคอลัมน์:\n", df.isnull().sum().sum(), "ค่า (รวม)")
    print("\nการกระจายของ Target (NObeyesdad):")
    print(df[TARGET].value_counts(), "\n")

    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 3-4: เตรียมข้อมูล และแบ่ง Train/Test")
    print("=" * 60)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} แถว | Test set: {X_test.shape[0]} แถว\n")

    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: สร้าง Pipeline (Preprocessing + SVM)")
    print("=" * 60)
    # ColumnTransformer: numeric -> StandardScaler, categorical -> OneHotEncoder
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
    ])
    print("สร้าง Pipeline สำเร็จ: [StandardScaler + OneHotEncoder] -> [SVM]\n")

    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: ปรับจูนพารามิเตอร์ด้วย GridSearchCV")
    print("=" * 60)
    param_grid = {
        "svm__C": [1, 10, 50],
        "svm__kernel": ["rbf", "linear"],
        "svm__gamma": ["scale", "auto"],
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("\nพารามิเตอร์ที่ดีที่สุด:", grid_search.best_params_)
    print(f"Cross-validation accuracy สูงสุด: {grid_search.best_score_:.4f}\n")

    # -----------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: ประเมินผลโมเดลด้วยชุดข้อมูล Test")
    print("=" * 60)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy บนชุดทดสอบ: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: บันทึกโมเดล (Pipeline เดียว ครบทั้ง preprocessing + SVM)")
    print("=" * 60)
    model_path = "model/obesity_svm_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"บันทึกโมเดลสำเร็จที่: {model_path}")
    print("โมเดลนี้เป็นไฟล์เดียว (single pipeline) พร้อมนำไปใช้ทำนายข้อมูลใหม่ได้ทันที")
    print("โดยไม่ต้องทำ preprocessing เพิ่มเติมเอง (เพราะรวมอยู่ใน Pipeline แล้ว)")


if __name__ == "__main__":
    main()
