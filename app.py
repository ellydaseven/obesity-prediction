# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:32:33 2023

@author: Hp
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from db import create_table, add_data, view_data

#loading the saved model

model = pickle.load(open('obesitymodel.sav', 'rb'))

st.set_page_config(layout="wide")

#sidebar nav code

with st.sidebar:
    
    select = option_menu('Menu',['Predictions','Analytics and Visuals','Records'], default_index=0)
    
    
# Prediction page

if(select == 'Predictions'):
    #Page title
    st.title("Obesity Prediction via Machine Learning") 
    st.write('Please answer the questions below')
    
    #input data
    with st.form(key='form1', clear_on_submit=True):
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
    
            with col1:
                age = st.number_input('What is your age?')
            with col2:
                height = st.number_input('What is your height? (in centimeters please)')
            with col3:
                weight = st.number_input('Enter your weight (in kilograms please')
            with col4:
                fastFood = st.selectbox('How often do you eat fast foods on a weekly basis?',('Never','Sometimes','Often','Always'))
    
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
    
            with col1:
                carbonDrink = st.selectbox('How often do you drink carbonated soft drinks (sodas, energy drinks)?', ('Always','Sometimes','Never'))
            with col2:
                alcohol = st.selectbox('How often do you consume alcohol?',('Frequently','Sometimes','Never'))
            with col3:
                exercise = st.selectbox('How often do you perform exercise activities?',('Never','Sometimes','Always'))
            with col4:
                sports = st.selectbox('How often do you engage in sports like football, running, swimming, etc?',('Never','Sometimes','Always'))
    
        
        submit = st.form_submit_button(label='Make prediction')
        
        if submit:
            # Encode the categorical data input
           
            #Fast food consumption encoding
            def fastFoodEncode(fastFood):
                if fastFood == 'Never':
                    fastFoodValue = 1
                elif fastFood == 'Sometimes':
                    fastFoodValue = 3
                elif fastFood == 'Often':
                    fastFoodValue = 2
                else:
                    fastFoodValue = 0
                return fastFoodValue
                            
            # Carbonated drinks consumption encoding
            def carbonDrinkEncode(carbonDrink):
                if carbonDrink == 'Always':
                    carbonDrinkValue = 0
                elif carbonDrink == 'Sometimes':
                    carbonDrinkValue = 2
                else:
                    carbonDrinkValue = 1
                return carbonDrinkValue
                
                            
            #Alcohol encoding
            def alcoholEncode(alcohol):
                if alcohol == 'Frequently':
                    alcoholValue = 0
                elif alcohol == 'Sometimes':
                    alcoholValue = 2
                else:
                    alcoholValue = 1
                return alcoholValue
                    
            # Exercise encoding
            def exerciseEncode(exercise):
                if exercise == 'Always':
                    exerciseValue = 0
                elif exercise == 'Sometimes':
                    exerciseValue = 2
                else:
                    exerciseValue = 1    
                return exerciseValue
            
            # Soprts encoding
            def sportsEncode(sports):
                if sports == 'Always':
                    sportsValue = 0
                elif sports == 'Sometimes':
                    sportsValue = 2
                else:
                    sportsValue = 1    
                return sportsValue
                    
                
                
            fast_food = fastFoodEncode(fastFood)
            carb = carbonDrinkEncode(carbonDrink)
            alc = alcoholEncode(alcohol)
            ex = exerciseEncode(exercise)
            sprt = sportsEncode(sports)

            if age < 5 or age > 100:
                st.warning("System use recommended for ages ranging from 5 to 100 years old")
            elif height < 60 or height > 240:
                st.warning("System use recommended for heights ranging from 60 to 240 centimeters")
            elif weight < 25 or weight > 200:
                st.warning("System use recommended for weights ranging from 25 to 200 kilograms")
            else:   
            
                #Capture the input, and put it into an numpy array
                input_data = [age, height, weight, fast_food, carb, alc, ex, sprt]
                pred_input = np.array([input_data])
                
                #Predict using the stacking model
                prediction = model.predict(pred_input)
                
                #Displaying the results
                def predict_result(pred):
                    if pred == 5:
                        st.error("Your obesity level is **Insufficient Weight**.You need to to gain more weight to stay healthy.")
                    elif pred == 0:
                        st.success("Your obesity level is **Normal Weight**. Keep up the good health and habits")
                    elif pred == 4:
                        st.warning("Your obesity level is **Overweight**")
                    elif pred == 1:
                        st.error("Your obesity Level is **Obese Class I**")
                    elif pred == 2:
                        st.error("Your obesity level is **Obese Class II**. Dangerous.")
                    else:
                        st.error("Your obesity level is **Obese Class III**. Highly dangerous.")
                        
                
                predict_result(prediction[0])
                
                def store_result(pred):
                    if pred == 5:
                        ob = "Insufficient weight"
                    elif pred == 0:
                        ob = "Normal weight"
                    elif pred == 4:
                        ob = "Overweight"
                    elif pred == 1:
                        ob = "Obese Class I"
                    elif pred == 2:
                        ob = "Obese Class II"
                    else:
                        ob = "Obese Class III"
                    return ob
                
                obesity = store_result(prediction[0])
            
                create_table()
                add_data(age, height, weight, fastFood, carbonDrink, alcohol, exercise, sports, obesity)
            
# Visualizations page

elif(select == 'Analytics and Visuals'):
    
    st.title('Dataset visualizations and Exploratory analysis')
    st.subheader('Here, we visualize the data in form of charts and diagrams')
            
       
    
    
    # Loading the data
    @st.cache_data
    def loadData(filename):
        df = pd.read_csv(filename)
        return df
        
    data = loadData('data_generated2.csv')

    # Distribution graphs
    st.markdown("""<style> .tab-heading{font-family:'Arial';
                                        color: #FF9633;
                                        }</style>""", unsafe_allow_html=True)
                
    st.markdown('<h6 class="tab-heading">The tabs below show the distribution of some predictors in our data</h6>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(['Weight', 'Height', 'Age'])
    
    # Weight distribution
    with tab1:
        fig1 = plt.figure(figsize=(16, 6))
        data['Weight'].hist() 
        plt.xlabel('Weight (kg)')
        
        plt.title("Distribution of weight in the dataset")
        st.pyplot(fig1)
        
    # Height distribution    
    with tab2:
        fig2 = plt.figure(figsize=(16, 6))
        data['Height'].hist() 
        plt.xlabel('Height (m)')
        
        plt.title("Distribution of height in the dataset")
        st.pyplot(fig2)
        
    # Age distribution    
    with tab3:
        fig3 = plt.figure(figsize=(16, 6))
        data['Age'].hist() 
        plt.xlabel('Age (yrs)')
        
        plt.title("Distribution of age in the dataset")
        st.pyplot(fig3)
        
    #Gender V Number of meals graph
    st.markdown('<h6 class="tab-heading">A graph of obesity levels per Gender: Bivariate analysis</h6>', unsafe_allow_html=True)
    fig4 = plt.figure(figsize=(16, 6))
    sns.countplot(x='Obesity Status', hue= 'Gender', data=data)
    plt.xticks(
        fontweight='light',
        fontsize='x-large'  
    )
    plt.legend()
    plt.title("Gender V Obesity Status")
    st.pyplot(fig4)
                
    st.markdown('<h4 class="tab-heading">Pie charts showing distributions of some categorical variables</h4>',unsafe_allow_html=True)
    
    tab4, tab5, tab6, tab7, tab8 = st.tabs(['Fast food consumption', 'Carbonated drinks consumption', 'Exercise', 'Sports activities','Obesity Status']) 
    
    with tab4:
        fig5 = plt.figure(figsize=(10, 6))
        data.groupby('Fast food consumption').size().plot(kind='pie', autopct='%.2f%%')
        plt.title("Fast food consumption - distribution of data")
        st.pyplot(fig5)
        
    with tab5:
        fig6 = plt.figure(figsize=(10, 6))
        data.groupby('Carbonated drinks consumption').size().plot(kind='pie', autopct='%.2f%%')
        plt.title("Carbonated drinks consumption - distribution of data")
        st.pyplot(fig6)
        
    with tab6:
        fig7 = plt.figure(figsize=(10, 6))
        data.groupby('Exercise').size().plot(kind='pie', autopct='%.2f%%')
        plt.title("Exercise frequency - distribution of data")
        st.pyplot(fig7)
        
    with tab7:
        fig8 = plt.figure(figsize=(10, 6))
        data.groupby('Sports activities').size().plot(kind='pie', autopct='%.2f%%')
        plt.title("Sports activity - distribution of data")
        st.pyplot(fig8)
               
        
    with tab8:
        fig9 = plt.figure(figsize=(10, 6))
        data.groupby('Obesity Status').size().plot(kind='pie', autopct='%.2f%%')
        plt.title("Obesity levels - distribution of data")
        st.pyplot(fig9)

else:
    st.title('Prediction Records')
    st.subheader('Here, we can see the records of real-time input and its predictions')
    
    records = view_data()
    
    df = pd.DataFrame(records, columns=['Age', 'Height', 'Weight', 'Fast food consumption', 'Carbonated drinks usage', 'Alcohol', 'Exercise', 'Sports', 'Obesity'])
    st.dataframe(df)
    
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    csv = convert_df(df)
    
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='prediction_data.csv',
        mime='text/csv',
    )
            
