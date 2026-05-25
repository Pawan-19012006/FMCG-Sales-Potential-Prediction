import numpy as np
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('sales_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
st.title("FMCG Sales Potential Prediction Dashboard")
st.write(
    "Estimate the potential sales performance of FMCG products "
    "based on product characteristics"
)

item_weight = st.number_input(
    "Enter Item Weight (in kgs)",
    min_value = 0.0
)

item_mrp = st.number_input(
    "Enter MRP (per kg)",
    min_value = 0.0
)

item_type = st.selectbox(
    "Select Item Type",
    ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
       'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
       'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
       'Breads', 'Starchy Foods', 'Others', 'Seafood']
)

fat_content = st.selectbox(
    "Enter Fat Category",
    ['Low Fat', 'Regular']
)

input_data = pd.DataFrame(
    np.zeros((1,len(feature_columns))),
    columns=feature_columns
)

input_data['Item_Weight'] = item_weight
input_data['Item_MRP'] = item_mrp
if fat_content == 'Regular':
    input_data['Item_Fat_Content_Regular'] = 1

col_name = f"Item_Type_{item_type}"

if col_name in input_data.columns:
        input_data[col_name] = 1

input_data['Item_Visibility'] = 0.07
input_data['Outlet_Size'] = 2
input_data['Outlet_Age'] = 15

if st.button("Predict Sales"):
    if item_weight <=0 or item_mrp<=0:
         st.error("Please Enter a Valid Weight or Amount")
    else:
        prediction = model.predict(input_data)[0]
        prediction = np.expm1(prediction)
        st.success(
              f"Predicted Sales: ₹{round(prediction, 2)}"
            )

