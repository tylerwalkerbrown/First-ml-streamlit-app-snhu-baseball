import requests
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from PIL import Image
import streamlit as st


logo = Image.open('logo.png')
st.image(logo, caption='Your Logo', use_column_width=True)


# Load the saved model and encoder
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

model = data['model']
encoder = data['encoder']

def predict(bat_avg, avg_HR, opp_error, game_avg, era_by_team, loc):
    loc_encoded = encoder.transform([loc])[0]
    new_data = pd.DataFrame({'bat_avg': [bat_avg],
                             'avg_HR': [avg_HR],
                             'opp_error': [opp_error],
                             '10_game_avg': [game_avg],
                             'avg_era_by_team': [era_by_team],
                             'Loc_encoded': [loc_encoded]})
    
    result = model.predict(new_data)[0]
    return result

def get_input():
    bat_avg = st.number_input('Batting average:', min_value=0.000, max_value=1.000, step=0.01, value=0.220)
    avg_HR = st.number_input('Average home runs:', min_value=0.000, max_value=5.000, step=0.01, value=0.500)
    opp_error = st.number_input('Opponent error:', min_value=0.000, max_value=1.000, step=0.01, value=0.200)
    game_avg = st.number_input('10-game average:', min_value=0.000, max_value=1.000, step=0.01, value=0.400)
    era_by_team = st.number_input('Average ERA by team:', min_value=0.000, max_value=10.000, step=0.01, value=1.600)
    loc = st.selectbox('Location:', ['vs', 'at'])
    result = predict(bat_avg, avg_HR, opp_error, game_avg, era_by_team, loc)
    return result

def main():
    st.title('Penman Baseball Win/Loss Predictor')
    st.markdown('Enter the data:')
    result = get_input()
    if result == 1:
        st.write('The predicted outcome is: **Win**')
    else:
        st.write('The predicted outcome is: **Loss**')

if __name__ == '__main__':
    main()

