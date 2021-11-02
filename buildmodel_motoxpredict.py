import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
df = pd.read_csv('konsumen.csv', delimiter=';',
                 skiprows=0, low_memory=False)

df = df.drop(columns=['BESAR DP', 'BESAR CICILAN',
             'LAMA CICILAN', 'KETERANGAN', 'SALES DATE', 'TANGGAL LAHIR'])
df.dropna(inplace=True)
list_1 = list(df.columns)

list_cate = []
for i in list_1:
    if df[i].dtype == 'object':
        list_cate.append(i)

le = LabelEncoder()
for i in list_cate:
    df[i] = le.fit_transform(df[i])

y = df[['TYPE MOTOR']]
X = df[['SMH DIGUNAKAN UNTUK', 'HOBI', 'PEKERJAAN']]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)
