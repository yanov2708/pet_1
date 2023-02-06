import pickle, requests, json
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from random import sample

from help_functions import return_heroes
from help_functions import list_to_df
from help_functions import load_model
from help_functions import load_df


heroes_list, dict_hero_id = return_heroes()
pickle_model = load_model()
loaded_classifier = pickle_model["sgd_model"]




def show_tabs():
    global tab1, tab2, tab3, tab4
    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "EDA", 'Hypothesis testing', 'Model'])

def show_predict_page():

    with tab1:
        st.title('Win prediction page')
        st.header('Fill match details')

        radiant_team, general_match_info, dire_team = st.columns(3, gap="large")


        with radiant_team:
            st.header('''Radiant''')
            radiant_score = st.number_input('Fill radiant score', step=1, value=39)
            st.write('The current number is ', radiant_score)

            radiant_heroes = st.multiselect('Сhoose your heroes', heroes_list, max_selections=5)
            #default=['Sniper', 'Phantom Assassin', 'Lina', 'Bloodseeker', 'Drow Ranger']


        with dire_team:
            st.header('''Dire''')
            dire_score = st.number_input('Fill dire score', step=1, value=43)
            st.write('The current number is ', dire_score)

            heroes_list_for_dire = list(set(heroes_list) - set(radiant_heroes))
            dire_heroes = st.multiselect('Сhoose your heroes', heroes_list_for_dire, max_selections=5)
            #default=['Alchemist', 'Lifestealer', 'Zeus', 'Phoenix', 'Pudge']


        with general_match_info:
            st.header("General info")
            duration_min = st.number_input('Match duration is :timer_clock:', step=1, value=29)
            st.write('Current duration in minutes is ', duration_min)
            duration_sec = duration_min * 60

            ok = st.button('Predict :rocket:')
            if ok:
                radiant_heroes_le = [dict_hero_id[hero] for hero in radiant_heroes]
                dire_heroes_le = [dict_hero_id[hero] for hero in dire_heroes]
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
        st.title('Some EDA of my data :chart_with_upwards_trend:')
        st.write('''When collecting data, I chose 23 features, but then I decided to use only 13+1(target), among which:
        
        \n
        'duration', 'radiant_score', 'dire_score', 
        'hero_radiant_1', 'hero_radiant_2', 'hero_radiant_3', 'hero_radiant_4', 'hero_radiant_5', 
        'hero_dire_1', 'hero_dire_2', 'hero_dire_3', 'hero_dire_4', 'hero_dire_5' ''')
        df = load_df()


        #1
        st.header('Some information about heroes :male_superhero:.')
        #1.1
        st.subheader('First, make sure that all existing heroes are included in our records.')
        code1 = '''list_with_all_heroes_from_df = df[['hero_radiant_1', 'hero_radiant_2', 'hero_radiant_3', 'hero_radiant_4', 'hero_radiant_5',
                                   'hero_dire_1', 'hero_dire_2', 'hero_dire_3', 'hero_dire_4', 'hero_dire_5']].to_numpy().ravel(order='F')
pd.Series(list_with_all_heroes_from_df, name='').nunique()
#this snippet returned - 123'''
        st.code(code1, language='python')
        #1.2
        st.subheader("Next, let's look at the top popular heroes for :green[Radiant] and :red[Dire] side.")
        ############  need update

        #2
        st.header('Description of our features, without hero columns.')
        st.dataframe(df.describe().T)

        #3
        st.header('Lets look at the number of wins each side :bar_chart:')
        nu_of_wins = Image.open('pictures/Number_of_wins.png')
        st.image(nu_of_wins)
        st.subheader('We see that the victories of the :green[Radiant] side are more, in the next section we will check if this is an accident or not.')

        #4
        st.header("Next, let's take a look at the distribution of the duration of matches in seconds, by the way, for training the model, I selected only turbo matches :fast_forward:.")
        dur_distrib = Image.open('pictures/Duration.png')
        st.image(dur_distrib)
        st.subheader('We see that the duration ranges from 15 minutes to 25')

        #5
        st.header("Next, let's look at the boxplot of each side's score.	:package:")
        box_score = Image.open('pictures/Boxplot.png')
        st.image(box_score)
        st.subheader('As we saw earlier, the :green[Radiant] won more often, it is natural that the median with the 1st and 3rd quartiles of their score is higher than that of the :red[Dire].')
        st.subheader('It is also worth paying attention to the fact that we have matches with a score below 10, which is suspicious, given the minimum match duration of 15 minutes')

        #6
        st.header("The last and less informative graph is:")
        math_per_day = Image.open('pictures/Number_of_collected_matches_per_day.png')
        st.image(math_per_day)
        st.subheader('Although the date of data collection may be useful, for example, this way you can understand which dota.patch the entries belong to')

    with tab3:
        st.title('Hypothesis testing')

    with tab4:
        st.title('About model')
