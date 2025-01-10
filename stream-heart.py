import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
df_clean = pd.read_csv('./dataset/heart_statlog_cleveland_hungary.csv', sep=';')
X = df_clean.drop("target", axis=1)
y = df_clean["target"]

# Melakukan oversampling menggunakan SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
with open('penyakit_jantung.pkl', 'wb') as f:
    pickle.dump(model, f) 
                          
# Memuat model dari file pickle
# model = pickle.load(open('penyakit_jantung.pkl', 'rb'))
try:
    with open('penyakit_jantung.pkl', 'rb') as f:
        model = pickle.load(f)  # Load the model object
except FileNotFoundError:
    st.error("Model file not found. Please train the model first.")
    st.stop()

df_final = X
df_final["target"] = y

# Streamlit app configuration
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",  # Heart emoji
)

# App title and explanation
st.title("Prediksi Penyakit Jantung")
st.write("Aplikasi ini membantu Anda memprediksi risiko penyakit jantung berdasarkan faktor-faktor yang Anda masukkan.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Usia', min_value=0, max_value=120, step=1)
    sex = st.selectbox('Jenis Kelamin', options=['Laki-laki', 'Perempuan'])
    chest_pain = st.selectbox('Tipe Nyeri Dada', options=[
        'Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    resting_bp = st.number_input('Tekanan Darah Istirahat (mmHg)', min_value=0, max_value=200)
    cholesterol = st.number_input('Kolesterol (mg/dL)', min_value=0, max_value=600)
    fasting_blood_sugar = st.selectbox('Gula Darah Puasa > 120 mg/dL', options=['Ya', 'Tidak'])

with col2:
    resting_ecg = st.selectbox('Hasil ECG Istirahat', options=['Normal', 'Abnormal ST-T', 'Hypertrophy'])
    max_heart_rate = st.number_input('Detak Jantung Maksimal (bpm)', min_value=0, max_value=220)
    exercise_angina = st.selectbox('Angina Saat Olahraga', options=['Ya', 'Tidak'])
    oldpeak = st.number_input('Depresi ST (Oldpeak)', min_value=0.0, max_value=10.0, step=0.1)
    ST_slope = st.selectbox('Kemiringan ST', options=['Up', 'Flat', 'Downsloping'])

# Map categorical input to numerical values
input_data = {
    'age': age,
    'sex': 1 if sex == 'Laki-laki' else 0,
    'chest pain type': {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-Anginal Pain': 3, 'Asymptomatic': 4}[chest_pain],
    'resting bp s': resting_bp,
    'cholesterol': cholesterol,
    'fasting blood sugar': 1 if fasting_blood_sugar == 'Ya' else 0,
    'resting ecg': {'Normal': 0, 'Abnormal ST-T': 1, 'Hypertrophy': 2}[resting_ecg],
    'max heart rate': max_heart_rate,
    'exercise angina': 1 if exercise_angina == 'Ya' else 0,
    'oldpeak': oldpeak,
    'ST slope': {'Up': 1, 'Flat': 2, 'Downsloping': 3}[ST_slope]
}
# inputs = ([[age, sex, chest_pain, resting_bp, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope]])
# Membuat DataFrame dari input pengguna
input_df = pd.DataFrame([input_data])

# Menampilkan DataFrame hasil input pengguna
st.header("User Input as DataFrame")
st.dataframe(input_df)

predict_btn = st.button("**Predict**", type="primary")

if predict_btn:
    
    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
        status_text.text(f"{i}% complete")
        bar.progress(i)
        time.sleep(0.01)
        if i == 100:
            time.sleep(1)
            status_text.empty()
            bar.empty()
            
    try:
        input_df_scaled = scaler.transform(input_df) # Scale the input
        prediction = model.predict(input_df_scaled)
        probability = model.predict_proba(input_df_scaled)[:, 1]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.stop()

    # Menampilkan hasil prediksi dan deskripsi hasilnya
    if prediction == 0:
        desc = 'Tidak Terindikasi Penyakit Jantung'
    elif prediction == 1:
        desc = 'Terindikasi Penyakit Jantung'
    # Menampilkan hasil prediksi dan deskripsi
    st.write("")
    st.subheader("prediction:")
    st.write(desc)

# Membuat sample file CSV
sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode("utf-8")

st.write("")
# Menambahkan tombol untuk mengunduh file CSV sample
st.download_button(
    "Download CSV Example",
    data=sample_csv,
    file_name="heart_statlog_cleveland_hungary.csv",
    mime="text/csv",
)
