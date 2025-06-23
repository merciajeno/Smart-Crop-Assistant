import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/Crop_recommendation.csv')
X=data.iloc[:,:-1]
Y = data.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42,shuffle=True)

random=RandomForestClassifier()
random.fit(X_train,Y_train)
accuracy = accuracy_score(random.predict(X_test),Y_test)
print(accuracy)
train_acc = random.score(X_train, Y_train)
test_acc = random.score(X_test, Y_test)
print(train_acc,test_acc)
# joblib.dump(random,'rf_crop_model.pkl')
# print('Model saved successfully')