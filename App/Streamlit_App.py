import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import pickle
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

st.sidebar.title('Menu')
menu = st.sidebar.radio('', ['Introduction', 'EDA', 'Prediction'])

if menu == 'Introduction':
    st.title('E-commerce Sales Predictor')
    st.write('Welcome to the ESP App! This app helps you to predict sales for a specific product at a given outlet.')
    st.image('./App/ecommerce.jpg')

    """
    The BigMart sales prediction dataset contains 2013's annual sales records for 1559 products across ten stores in different cities. Such vast data can reveal insights about apparent customer preferences as a specific product and store attributes have been defined in the dataset. 

    **Data Dictionary:**
    * **item_identifier:** Unique Identification number for particular items
    * **item_weight:** Weight of the items
    * **item_fat_content:** fat content in the item such as low fat and regular fat
    * **item_visibility:** visibility of the product in the outlet
    * **item_type:** category of the product such as Dairy, Soft Drink, Household, etc
    * **item_mrp:** Maximum retail price of the product
    * **outlet_identifier:** unique identification number for particular outlets    
    * **outlet_establishment_year:** the year in which the outlet was established
    * **outlet_size:** the size of the outlet, such as small, medium, and high
    * **outlet_location_type:** location type in which the outlet is located, such as Tier 1, 2 and 3
    * **outlet_type:** type of the outlet such as grocery store or supermarket
    * **item_outlet_sales:** overall sales of the product in the outlet
    """
    st.write(' ')
    st.write('**Sales Data:**')
    df = pd.read_csv('./Data/Train.csv')
    st.dataframe(df)

elif menu == 'EDA':
    st.title('Exploratory Data Analysis')
    st.header('1. Univariate Analysis')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.1 Distribution of Outlet Size</p>", unsafe_allow_html=True)
    st.image('./Output/Outlet_Size.png')
    st.markdown("<div style='text-align:center;'>Most of the outlets are Small (56.3%) followed by Medium (32.8%) and Large (10.9%).</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.2 Distribution of Outlet Type</p>", unsafe_allow_html=True)
    st.image('./Output/Outlet_Type.png')
    st.markdown("<div style='text-align:center;'>Most of the outlet types are Supermarket Type 1 (65.4%) and the least are Supermarket Type 2 (10.9%).</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.3 Timeline of Outlet Establishments</p>", unsafe_allow_html=True)
    st.image('./Output/Outlet_Establishment_Year.png')
    st.markdown("<div style='text-align:center;'>Two outlets exists since the year 1985 and most recent one is established in 2009.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.4 Distribution of Item Categories</p>", unsafe_allow_html=True)
    st.image('./Output/Item_Categories.png')
    st.markdown("<div style='text-align:center;'>Majority of Items belong to Food (71.9%) category followed by Non-Consumable (18.8%) and Drinks (9.4%).</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.5 Popular Item Categories</p>", unsafe_allow_html=True)
    st.image('./Output/Item_Type.png')
    st.markdown("<div style='text-align:center;'>Fruits and Vegetables, Snack Foods and Household Items are the top 3 food items.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.6 Distribution of Item Fat Content</p>", unsafe_allow_html=True)
    st.image('./Output/Item_Fat_Content.png')
    st.markdown("<div style='text-align:center;'>Most of the items have Low Fat Content compared to Regular.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.7 Distribution of Item MRP Categories</p>", unsafe_allow_html=True)
    st.image('./Output/Item_MRP_Categories.png')
    st.markdown("<div style='text-align:center;'>Most of items are priced between 70 USD and 202 USD.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>1.8 Distribution of Target Label (Item Outlet Sales)</p>", unsafe_allow_html=True)
    st.image('./Output/Item_Outlet_Sales.png')
    st.markdown("<div style='text-align:center;'>The target label is right skewed and most of the Outlet Sales range from 0 to 4000 USD.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.header('2. Bivariate Analysis')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>2.1 Distribution of Outlet Sales w.r.t. Fat Content</p>", unsafe_allow_html=True)
    st.image('./Output/ItemFatContent_ItemOutletSales.png')
    st.markdown("<div style='text-align:center;'>Outlet Sales are not affected by the Fat Content present in Items.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>2.2 Distribution of Outlet Sales w.r.t. Item Visibility</p>", unsafe_allow_html=True)
    st.image('./Output/ItemVisibility_ItemOutletSales.png')
    st.markdown("<div style='text-align:center;'>Item Visibility is effective till 0.2, beyond that it doesn't have any impact on Outlet Sales.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>2.3 Distribution of Outlet Sales for each Outlet</p>", unsafe_allow_html=True)
    st.image('./Output/OutletID_ItemOutletSales.png')
    st.markdown("<div style='text-align:center;'>Outlet \'OUT027\' has the highest sales and \'OUT010\' & \'OUT019\' has the least sales.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>2.4 Distribution of Outlet Sales w.r.t. Outlet Size</p>", unsafe_allow_html=True)
    st.image('./Output/OutletSize_ItemOutletSales.png')
    st.markdown("<div style='text-align:center;'>Outlet Size doesn't have much impact on Outlet Sales.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>2.5 Distribution of Outlet Sales w.r.t. Outlet Type</p>", unsafe_allow_html=True)
    st.image('./Output/OutletType_ItemOutletSales.png')
    st.markdown("<div style='text-align:center;'>\'Supermarket Type3\' has the highest sales and \'Grocery Store\' has the lowest sales.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')

    st.header('3. Multivariate Analysis')

    st.markdown("<p style='font-size:20px; font-weight:bold;'>Correlation Heatmap</p>", unsafe_allow_html=True)
    st.image('./Output/Heatmap.png')
    st.markdown("<div style='text-align:center;'>Item MRP and Item Outlet Sales has positive correlation. Item Visibility and Item Outlet Sales has negative correlation.</div>", unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')


else:
    st.header('Input Features')
    item_weight = st.number_input('Item Weight (lbs)', min_value=2.0, 
                                  max_value=30.0, value=13.0)
    
    item_categories = st.radio('Item Category', ['Drink', 'Food', 'Non-Consumable'], horizontal=True)
    
    item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular Fat'])
    
    item_visibility = st.number_input('Item Visibility', min_value=0.0, 
                                      max_value=5.0, value=0.06)
    item_mrp = st.slider('Item MRP', min_value=10.0, max_value=350.0, value=140.0)
    
    item_mrp_category = st.radio('Item MRP Category Type', ['First (0-69)', 'Second (70-135)', 
                         'Third (136-202)', 'Fourth (203-270)'], horizontal=True)
    
    item_mrp_category_mapping = {'First (0-69)':0, 'Second (70-135)':2, 'Third (136-202)':3, 'Fourth (203-270)':1}
    
    outlet_identifier = st.selectbox('Outlet ID', ['OUT010', 'OUT045', 'OUT017', 
                                                   'OUT046', 'OUT035', 'OUT019', 
                                                   'OUT049', 'OUT018', 'OUT027', 
                                                   'OUT013'])
    outlet_identifier_mapping = {'OUT010':0, 'OUT013':1, 'OUT017':2, 'OUT018':3, 'OUT019':4, 
                                 'OUT027':5, 'OUT035':6, 'OUT045':7, 'OUT046':8, 'OUT049':9}

    outlet_age = st.slider('Outlet Age', min_value=0, max_value=50, value=15)
    
    outlet_size = st.radio('Outlet Size', ['Small', 'Medium', 'Large'], horizontal=True)
    
    outlet_location_type = st.radio('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'], 
                                    horizontal=True)
    
    outlet_type = st.radio('Outlet Type', ['Grocery Store', 'Supermarket Type1', 
                                           'Supermarket Type2', 'Supermarket Type3'], 
                                           horizontal=True)
    outlet_type_mapping = {'Grocery Store':0, 'Supermarket Type1':1, 
                           'Supermarket Type2':2, 'Supermarket Type3':3}
    df = pd.DataFrame(
        {'Item_Weight':[item_weight], 
         'Item_Fat_Content':[0 if item_fat_content=='Low Fat' else 1],
         'Item_Visibility':[item_visibility],
         'Item_MRP':[item_mrp],
         'Outlet_Identifier':outlet_identifier_mapping[outlet_identifier],
         'Outlet_Size':[2 if outlet_size=='Small' else 1 if outlet_size=='Medium' else 0],
         'Outlet_Location_Type':[0 if outlet_location_type=='Tier 1' else 1 if outlet_location_type=='Tier2' else 2],
         'Outlet_Type':outlet_type_mapping[outlet_type],
         'Item_Categories':[0 if item_categories=='Drink' else 1 if item_categories=='Food' else 2],
         'Outlet_Age':[outlet_age],
         'Item_MRP_Categories':item_mrp_category_mapping[item_mrp_category]
         }
    )

    st.write('User Input Values:')
    st.write(df)
    with open('./Pickle Files/Scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    df_scaled = scaler.transform(df)
    # st.write(df_scaled)

    if st.button('Predict'):
        with open('./Pickle Files/GBM_Model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        predicted_sales = model.predict(df_scaled)
        st.write(f'Predicted Outlet Sales for the Item: ${predicted_sales}')
