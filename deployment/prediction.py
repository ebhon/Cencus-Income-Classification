import streamlit as st
import pandas as pd
import pickle

# Load  model
model_path = "pipeline.pkl" 
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def run_modelling(user_input):
    prediction = model.predict(user_input)
    return prediction

# Function to get user input from the sidebar
def get_user_input():
    st.sidebar.header("Input Parameters")

    # Use a unique key by appending a suffix or index
    age = st.sidebar.number_input("Usia", min_value=0, max_value=120, value=25, key='age_input')
    
    # Select boxes with options for education, marital status, and occupation
    workclass = st.sidebar.selectbox("Jenis Pekerjaan", ["Swasta", "Wiraswasta (Tidak Terdaftar)", "Wiraswasta (Terdaftar)", 
                                                 "Pemerintah Pusat", "Pemerintah Lokal", "Pemerintah Daerah", 
                                                 "Tanpa Bayaran", "Belum Pernah Bekerja"], key='workclass_input')
    education = st.sidebar.selectbox("Pendidikan", ['Lulusan SMA', 'Sebagian Kuliah', 'Sarjana', 
                                                        'Magister', 'Diploma Vokasi', 'SMA (Kelas 11)', 
                                                        'Diploma Akademik', 'SMA (Kelas 10)', 
                                                        'SMP (Kelas 7-8)', 'Sekolah Profesional', 
                                                        'SMP (Kelas 9)', 'SMA (Kelas 12)', 'Doktor', 
                                                        'SD (Kelas 5-6)', 'SD (Kelas 1-4)', 
                                                        'TK (Taman Kanak-Kanak)'], key='education_input')
    fnlwgt = st.sidebar.number_input("Bobot Akhir", value=0, key='fnlwgt_input')
    marital_status = st.sidebar.selectbox("Status Perkawinan", ['Menikah', 'Belum Pernah Menikah', 
                                                                    'Bercerai', 'Berpisah', 'Duda/Janda', 
                                                                    'Menikah (Pasangan Tidak Ada)', 
                                                                    'Menikah (Pasangan di Militer)'], key='marital_status_input')
    occupation = st.sidebar.selectbox("Pekerjaan", ['Profesional', 'Perbaikan Kerajinan', 
                                                        'Eksekutif/Manajerial', 'Administrasi/Klerikal', 
                                                        'Penjualan', 'Layanan Lain', 
                                                        'Operator Mesin/Inspeksi', 'Transportasi/Pengemudi', 
                                                        'Pembersih/Tenaga Kasar', 'Pertanian/Perikanan', 
                                                        'Dukungan Teknis', 'Layanan Perlindungan', 
                                                        'Pelayan Rumah Tangga', 'Angkatan Bersenjata'], key='occupation_input')
    relationship = st.sidebar.selectbox("Hubungan", ["Suami","Tidak Dalam Keluarga", "Anak Sendiri", "Unmarried", "Istri", "Kerabat Lain"], key='relationship_input')
    race = st.sidebar.selectbox("Ras", ["Caucasian (Putih)" , "Afrika (Hitam)", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Lain-lain"], key='race_input')
    gender = st.sidebar.selectbox("Jenis kelamin", ['Perempuan', 'Laki-Laki'], key='gender_input')
    capital_gain = st.sidebar.number_input("Keuntungan Modal", value=0, key='capital_gain_input')
    capital_loss = st.sidebar.number_input("Kerugian Modal", value=0, key='capital_loss_input')
    hours_per_week = st.sidebar.number_input("Jam Kerja per Minggu", value=40, key='hours_per_week_input')
    native_country = st.sidebar.selectbox("Negara Asal", ['United-States', 'Cambodia', 'England', 
                                                            'Puerto-Rico', 'Canada', 'Germany', 
                                                            'India', 'Japan', 'Greece', 'South', 
                                                            'China', 'Cuba', 'Iran', 'Honduras', 
                                                            'Philippines', 'Italy', 'Poland', 
                                                            'Columbia', 'Mexico', 'Portugal', 
                                                            'South Africa', 'Taiwan', 'Thailand', 
                                                            'Yugoslavia'], key='native_country_input')

    # Create a DataFrame from the inputs 
    user_input = pd.DataFrame({
            'usia': [age],  
            'jenis_pekerjaan': [workclass],  
            'bobot_akhir': [fnlwgt],  
            'pendidikan': [education],  
            'nomor_pendidikan': [12],  
            'status_perkawinan': [marital_status],  
            'pekerjaan': [occupation],  
            'hubungan': [relationship], 
            'ras': [race], 
            'jenis_kelamin': [gender],  
            'keuntungan_modal': [capital_gain],  
            'kerugian_modal': [capital_loss],  
            'jam_kerja': [hours_per_week],  
            'negara_asal': [native_country]  
    })

    return user_input
