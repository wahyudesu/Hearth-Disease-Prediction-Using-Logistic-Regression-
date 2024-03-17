# Import necessary libraries
import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression

# Logo [optionnal]
st.image("gambar anime.webp", use_column_width=True)

# create streamlit interface, asome info about the app
'''
st.write("""
         ## In just a few seconds, you can calculate your risk of developing heart disease!
      the app is built based on the 2020 annual CDC survey data of 400k  adults related to their health status,
        using machine learning algorithm called logistic regression with an accuracy of 88%.
        To learn more about the data, check the following link: [Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease). 
        If you are interested to check my code, check my github using the following link: [Github](https://github.com/lamisghoualmi/App-Personal-Key-Indicators-of-Heart-Disease). Note: this results are not equivalent to a medical diagnosis!  
         """)
'''
st.write("""
         ### To predict your heart disease status:
         ###### 1- Masukkan parameters sesuai dengan keadaan kesehatanmu di sidebar samping ini.
         ###### 2- Penced "Predict" button dan tunggu hasilnya.
         """)

st.subheader("Created by Wahyu Ikbal Maulana From SDT B")

# st.write(BMIdata)

# Sidebar input
# -------------------------------------------------------------------------
st.sidebar.title('Jawab pertanyaan-pertanyaan berikut')

BMI=st.sidebar.selectbox("Berapa BMI mu", ("Normal weight BMI  (18.5-25)", 
                             "Kurus BMI (< 18.5)" ,
                             "Gemuk dikit BMI (25-30)",
                             "Obesitas BMI (> 30)"))
Age=st.sidebar.selectbox("Pilih rentang tahun", 
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

Race=st.sidebar.selectbox("Pilih ras kamu", ("Asian", 
                             "Nigga" ,
                             "Hispanic",
                             "American Indian/Alaska Native",
                             "Sipit",
                             "Other"
                             ))

Gender=st.sidebar.selectbox("Pilih jenis kelamin", ("Perempuan", 
                             "Laki-laki" ))
Smoking = st.sidebar.selectbox("Have you smoked more than 100 cigarettes in"
                          " your entire life ?)",
                          options=("No", "Yes"))
alcoholDink = st.sidebar.selectbox("Apakah kamu sering minum alkohol?", options=("No", "Yes"))
stroke = st.sidebar.selectbox("Pernah stroke?", options=("No", "Yes"))

sleepTime = st.sidebar.number_input("Berapa jam kamu tidur dalam sehari?", 0, 24, 7) 

genHealth = st.sidebar.selectbox("Kesehatanmu?",
                             options=("Good","Excellent", "Fair", "Very good", "Poor"))

physHealth = st.sidebar.number_input("Berapa skor nilai fisikmu? (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
mentHealth = st.sidebar.number_input("Skor Mental health  (Excelent: 0 - Very bad: 30)"
                                 , 0, 30, 0)
physAct = st.sidebar.selectbox("Sering olahraga?"
                           , options=("No", "Yes"))

diffWalk = st.sidebar.selectbox("Apakah kamu mengalami kesulitan untuk berjalan"
                            " atau naik tangga?", options=("No", "Yes"))
diabetic = st.sidebar.selectbox("Pernah diabetes?",
                           options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
asthma = st.sidebar.selectbox("Punya riwayat penyakit asma?", options=("No", "Yes"))
kidneyDisease= st.sidebar.selectbox("Punya riwayat penyakit ginjal?", options=("No", "Yes"))
skinCancer = st.sidebar.selectbox("Punya riwayat penyakit kanker??", options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke": [stroke],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Gender": [Gender],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
 })

# Mapping the data as explained in the script above
dataToPredic.replace("Underweight BMI (< 18.5)",0,inplace=True)
dataToPredic.replace("Normal weight BMI  (18.5-25)",1,inplace=True)
dataToPredic.replace("Overweight BMI (25-30)",2,inplace=True)
dataToPredic.replace("Obese BMI (> 30)",3,inplace=True)

dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",0,inplace=True)
dataToPredic.replace("18-24",0,inplace=True)
dataToPredic.replace("25-29",1,inplace=True)
dataToPredic.replace("30-34",2,inplace=True)
dataToPredic.replace("35-39",3,inplace=True)
dataToPredic.replace("40-44",4,inplace=True)
dataToPredic.replace("45-49",5,inplace=True)
dataToPredic.replace("50-54",6,inplace=True)
dataToPredic.replace("55-59",7,inplace=True)
dataToPredic.replace("60-64",8,inplace=True)
dataToPredic.replace("65-69",9,inplace=True)
dataToPredic.replace("70-74",10,inplace=True)
dataToPredic.replace("75-79",11,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)


dataToPredic.replace("No, borderline diabetes",2,inplace=True)
dataToPredic.replace("Yes (during pregnancy)",3,inplace=True)


dataToPredic.replace("Excellent",0,inplace=True)
dataToPredic.replace("Good",1,inplace=True)
dataToPredic.replace("Fair",2,inplace=True)
dataToPredic.replace("Very good",3,inplace=True)
dataToPredic.replace("Poor",4,inplace=True)


dataToPredic.replace("Sipit",0,inplace=True)
dataToPredic.replace("Other",1,inplace=True)
dataToPredic.replace("Nigga",2,inplace=True)
dataToPredic.replace("Hispanic",3,inplace=True)
dataToPredic.replace("Asian",4,inplace=True)
dataToPredic.replace("American Indian/Alaskan Native",4,inplace=True)


dataToPredic.replace("Perempuan",0,inplace=True)
dataToPredic.replace("Laki-laki",1,inplace=True)

# Load the previously saved machine learning model
filename='finalized_model.sav'
loaded_model= pickle.load(open(filename, 'rb'))
Result=loaded_model.predict(dataToPredic)
ResultProb= loaded_model.predict_proba(dataToPredic)
ResultProb1=round(ResultProb[0][1] * 100, 2)

 # Calculate the probability of getting heart disease
if st.button('PREDICT'):
 # st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
 if (ResultProb1>30):
  st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
 else:
  st.write('You have a', ResultProb1, '% chance of getting a heart disease' )

  
