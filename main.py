import pandas as pd
df= pd.read_csv('collegePlace.csv')

#Data Processing
X=df[['Age','Internships','CGPA']]
Y=df['PlacedOrNot']

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
cor=df.select_dtypes(include='number').corr()
sns.heatmap(cor,annot=True)
plt.show()

#Model training
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(Y_test,Y_pred)
print(accuracy)

#Function
def Placement(Age,Intership,CGPA):
    prediction=model.predict([[Age,Intership,CGPA]])
    return 'Yes'if prediction[0]==1 else 'No'

#Streamlit Interface
import streamlit as st
st.title(" **🎓WHILE YOU BE PLACED**")
Age=st.number_input("Enter your Age:")
Intership=st.number_input("Enter the number of Interships you have attended:")
CGPA=st.number_input("Enter your CGPA:")

button=st.button("Predict")

if button:
    result = Placement(Age, Intership, CGPA)
    st.success(f"Will you be placed? **{result}**")