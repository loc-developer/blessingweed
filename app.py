import streamlit as st
import joblib
import os
import cv2
import numpy as np
import pandas as pd

IMG_PATH = os.path.join(os.getcwd(), 'images/')
CLASS_NAMES = ['Chinese Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negative']
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
test = pd.read_csv('new_test.csv')
model = joblib.load('xgb.pkl')

def flatten(file: str):
    test_dat = []
    img = cv2.imread(IMG_PATH + file)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.resize(new_img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    test_dat.append(new_img.reshape(224*224))
    test_dat = np.array(test_dat)
    return test_dat

st.title('Weed Detection Interface')

sel = st.button('Select a random Image from the test dataset and predict the class of the image')
if sel:
    row = test.sample()
    label = row['Label'].values[0]
    classe = CLASS_NAMES[CLASSES.index(label)]
    img_view = cv2.imread('images/' + row['Filename'].values[0])
    st.image(img_view, caption=classe, clamp=True)

    img = flatten(row['Filename'].values[0])

    pred = model.predict(img)[0]
    pred_label = CLASS_NAMES[CLASSES.index(pred)]

    st.write(f"Original Class of Plant shown is {label} and the class name is {classe}")
    st.write(f"The model predicted class {pred} which gave a class name of {pred_label} with an accuracy of 63%")