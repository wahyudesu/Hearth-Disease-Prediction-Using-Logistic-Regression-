# Import necessary libraries
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df=pd.read_csv("heart_2020_cleaned.csv")
#------------------Removing duplicated samples-------
duplicateObser = df[df.duplicated()]
LabelsDupObser=duplicateObser.axes[0].tolist()
print('Number of duplicated observations:', duplicateObser.shape[0])
df=df.drop_duplicates()
print(df.shape)

#------------------Data mapping ------------------
'''
Data mapping adalah telnik yang biasa digunakan untuk mengubah beberapa type
data dengan mengubah format supaya mudah dipakai di algoritma machine learning
untuk meningkatkan performa model. Seperti contoh 
'''
#Data mapping is a technique used to transform different types of data 
#to a common format that is suitable for machine learning models to improve 
#the model's performance. For example, BMI, a continuous variable, can be mapped
#to four categories: Normal weight BMI, Underweight BMI, Overweight BMI, and
#Obese. This mapping helps the model to better understand the patterns in the
#data and make more accurate predictions.

df.replace("Yes",1,inplace=True)
df.replace("No",0,inplace=True)

target=df["HeartDisease"]
df.drop(["HeartDisease"], axis=1, inplace=True)
df.AgeCategory.unique()
df.replace("18-24",0,inplace=True)
df.replace("25-29",1,inplace=True)
df.replace("30-34",2,inplace=True)
df.replace("35-39",3,inplace=True)
df.replace("40-44",4,inplace=True)
df.replace("45-49",5,inplace=True)
df.replace("50-54",6,inplace=True)
df.replace("55-59",7,inplace=True)
df.replace("60-64",8,inplace=True)
df.replace("65-69",9,inplace=True)
df.replace("70-74",10,inplace=True)
df.replace("75-79",11,inplace=True)
df.replace("80 or older",13,inplace=True)

df.Diabetic.unique()
df.replace("No, borderline diabetes",2,inplace=True)
df.replace("Yes (during pregnancy)",3,inplace=True)

df.GenHealth.unique()
df.replace("Excellent",0,inplace=True)
df.replace("Good",1,inplace=True)
df.replace("Fair",2,inplace=True)
df.replace("Very good",3,inplace=True)
df.replace("Poor",4,inplace=True)

df.Race.unique()
df.replace("White",0,inplace=True)
df.replace("Other",1,inplace=True)
df.replace("Black",2,inplace=True)
df.replace("Hispanic",3,inplace=True)
df.replace("Asian",4,inplace=True)
df.replace("American Indian/Alaskan Native",4,inplace=True)

df.Sex.unique()
df.replace("Female",0,inplace=True)
df.replace("Male",1,inplace=True)

df['BMI'].mask(df['BMI']  < 18.5, 0, inplace=True)
df['BMI'].mask(df['BMI'].between(18.5,25), 1, inplace=True)
df['BMI'].mask(df['BMI'].between(25,30), 2, inplace=True)
df['BMI'].mask(df['BMI']  > 30, 3, inplace=True)

# Split the data into training and testing
X_train,X_test,y_train,y_test = train_test_split(df,target,test_size=50,random_state=2)

# Train a logistic regression model on the training set
LogRegModel=LogisticRegression()
LogRegModel.fit(X_train, y_train)

# Save the model using pickle
with open('LogRegModel.pkl', 'wb') as f:
    pickle.dump(LogRegModel, f)
