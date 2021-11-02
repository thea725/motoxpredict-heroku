import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# MotoXPredict
Motorcycle prediction app Mark00
""")

st.sidebar.header("Masukan Informasi")


def input_form():
    motor = st.sidebar.number_input("Motor", 0, 50)
    hobi = st.sidebar.number_input("Hobi", 0, 50)
    pekerjaan = st.sidebar.number_input("Pekerjaan", 0, 50)
    data = {"motor": motor,
            "hobi": hobi,
            "pekerjaan": pekerjaan,
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = input_form()
df = df[:1]

load_clf = pickle.load(open("motor_clf.pkl", "rb"))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader("Prediction")
st.write(prediction)

st.subheader("Prediction Probability")
st.write(prediction_proba)
