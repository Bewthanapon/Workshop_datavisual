
import numpy as np
import pandas as pd
import streamlit as st
import pickle

st.write("""
# This app predict Charges 
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/Bewthanapon/Workshop_datavisual/blob/main/Insurance_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('age', 18,64,35)
        bmi = st.sidebar.slider('bmi', 16,53,28)
        sex = st.sidebar.selectbox('sex',('male','female'))
        children = st.sidebar.selectbox('children',('0','1','2','3','4','5'))
        smoker = st.sidebar.selectbox('smoker',('no','yes'))
        region = st.sidebar.selectbox('region',('northeast','northwest','southeast','southwest'))
        data = {'age': age,
                'bmi': bmi,
                'sex': sex,
                'children': children,
                'smoker': smoker,
                'region': region}
        features = pd.DataFrame(data, index=[0])
        return features
    input_data = user_input_features()

save_model = pickle.load(open('insurance_model.pickle', 'rb'))

model = save_model[0]
one_hot_encoder = save_model[1]
feature_name = save_model[2]
numeric = save_model[3]
onehot = save_model[4]

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_data)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_data)

# input_data = pd.DataFrame(np.array([[age, sex, bmi ,children, smoker, region]]), 
#                              columns=feature_name)

one_hot_feature = []
for i, feature in enumerate(onehot):
    for cate in one_hot_encoder.categories_[i]:
        one_hot_feature_name = str(feature) + '_' + str(cate)
        one_hot_feature.append(one_hot_feature_name)

input_data[one_hot_feature] = one_hot_encoder.transform(input_data[onehot])
input_data.drop(onehot, axis=1, inplace=True)



st.subheader('Prediction')
pred_input = model.predict(input_data)
Ans = float(pred_input)
Ans