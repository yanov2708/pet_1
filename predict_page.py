import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import json
#import joblib

url_heroes = 'https://api.opendota.com/api/heroes'
response_heroes = requests.get(url_heroes)
try:
    data_heroes = json.loads(response_heroes.content.decode("utf-8"))
    print('Amount dota heroes:', len(data_heroes))
    dict_id_hero = {data_hero['localized_name']: data_hero['id'] for data_hero in data_heroes}
except:
    print('No json data_heroes returned', '\n')
heroes_list = dict_id_hero.keys()


#func list to df for predict
def list_to_df(lst):
  cols = ['duration', 'radiant_score', 'dire_score',
          'hero_radiant_1', 'hero_radiant_2', 'hero_radiant_3', 'hero_radiant_4', 'hero_radiant_5',
          'hero_dire_1', 'hero_dire_2', 'hero_dire_3', 'hero_dire_4', 'hero_dire_5']
  dct = dict(zip(cols, lst))
  return pd.DataFrame(data=dct, index=[0])


#Load model
def load_model():
    with open('sgd_model.pkl', 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

pickle_model = load_model()
loaded_classifier = pickle_model["sgd_model"]

#best_sgd_model_joblib = joblib.load('sgd.joblib')




def show_tabs():
    global tab1, tab2, tab3
    tab1, tab2, tab3 = st.tabs(["Predict", "EDA", 'Model'])

def show_predict_page():

    with tab1:
        st.title('Win prediction page')
        st.header('Fill match details')

        radiant_team, general_match_info, dire_team = st.columns(3, gap="large")


        with radiant_team:
            st.header('''Radiant''')
            radiant_score = st.number_input('Fill radiant score', step=1, value=39)
            st.write('The current number is ', radiant_score)

            radiant_heroes = st.multiselect('Сhoose your heroes', heroes_list, max_selections=5, default=['Sniper', 'Phantom Assassin', 'Lina', 'Bloodseeker', 'Drow Ranger'])



        with dire_team:
            st.header('''Dire''')
            dire_score = st.number_input('Fill dire score', step=1, value=43)
            st.write('The current number is ', dire_score)

            heroes_list_for_dire = list(set(heroes_list) - set(radiant_heroes))
            dire_heroes = st.multiselect('Сhoose your heroes', heroes_list_for_dire, max_selections=5, default=['Alchemist', 'Lifestealer', 'Zeus', 'Phoenix', 'Pudge'])



        with general_match_info:
            st.header("General info")
            duration_min = st.number_input('Match duration is :timer_clock:', step=1, value=29)
            st.write('Current duration in minutes is ', duration_min)
            duration_sec = duration_min * 60

            ok = st.button('Predict :rocket:')
            if ok:
                radiant_heroes_le = [dict_id_hero[hero] for hero in radiant_heroes]
                dire_heroes_le = [dict_id_hero[hero] for hero in dire_heroes]
                data_for_pred = [duration_min*60, radiant_score, dire_score]
                data_for_pred.extend(radiant_heroes_le)
                data_for_pred.extend(dire_heroes_le)

                X = list_to_df(data_for_pred)
                win_pred = loaded_classifier.predict(X)  #loaded_classifier
                if win_pred == 1:
                    win_pred = 'Radiant'
                else:
                    win_pred = 'Dire'
                st.subheader(f'{win_pred} win! :tada::tada::tada:')

    with tab2:
        st.title('Some EDA of my dataset :bar_chart:')


    with tab3:
        st.title('About model')
