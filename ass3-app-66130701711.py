
import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import Perceptron

model = pickle.load(open('model_per-66130701711.sav', 'rb'))

st.title("Iris Species Prediction using Perceptrin")

x1 = st.slider('Seclect Input1', 0.0, 10.0, 3.0)
x2 = st.slider('Seclect Input2', 0.0, 10.0, 5.0) 
x3 = st.slider('Seclect Input3', 0.0, 10.0, 4.0) 
x4 = st.slider('Seclect Input4', 0.0, 10.0, 7.0)

xnew = np.array([x1,x2,x3,x4]) .reshape(1,-1)

xnew

pred = model.predict(xnew)
                    
st.write("## Prediction Result:")
st.write('Species:', pred[0])          
