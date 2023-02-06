import pickle, requests, json
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
#from random import sample

from help_functions import return_heroes
from help_functions import list_to_df
from help_functions import load_model
from help_functions import load_df_5f
from help_functions import load_df_15f
from help_functions import return_hero_frequency
from help_functions import return_count_victories

heroes_list, dict_hero_id, dict_id_hero = return_heroes()
pickle_model = load_model()
loaded_classifier = pickle_model["sgd_model"]




def show_tabs():
    global tab1, tab2, tab3
    tab1, tab2, tab3 = st.tabs(["Predict", "EDA", 'Hypothesis testing'])

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
        df = load_df_5f()


        #1
        st.header('Some information about heroes :male_superhero:.')
        #1.1
        st.subheader('First, make sure that all existing heroes are included in our records.')
        code1 = '''list_with_all_heroes_from_df = df[['hero_radiant_1', 'hero_radiant_2', 'hero_radiant_3', 'hero_radiant_4', 'hero_radiant_5',
                                   'hero_dire_1', 'hero_dire_2', 'hero_dire_3', 'hero_dire_4', 'hero_dire_5']].to_numpy().ravel(order='F')
pd.Series(list_with_all_heroes_from_df, name='').nunique()
#this snippet returned: 123'''
        st.code(code1, language='python')
        #1.2
        st.subheader("Next, let's look at the five most and least popular heroes.")
        head_heroes, tail_heroes = return_hero_frequency()
        popular, not_popular, = st.columns(2, gap='large')
        with popular:
            st.markdown('''### Popular 	:small_red_triangle: ''')
            st.dataframe(head_heroes, use_container_width=True)
        with not_popular:
            st.markdown('''### Less popular :small_red_triangle_down: ''')
            st.dataframe(tail_heroes, use_container_width=True)

        #2
        st.header('Description of our features, without hero columns.')
        st.dataframe(df.describe().T)

        #3
        st.header('Lets look at the number of wins each side :bar_chart:')
        nu_of_wins = Image.open('pictures/Number_of_wins.png')
        st.image(nu_of_wins)
        st.subheader('We see that the victories of the :green[Radiant] side are more, in the next section we will check if this is an accident or not.')

        #4
        st.header("Next, let's take a look at the distribution of the duration of matches in seconds, by the way, I selected only turbo matches :fast_forward:.")
        dur_distrib = Image.open('pictures/Duration.png')
        st.image(dur_distrib)
        st.subheader('We see that the duration ranges from 15 minutes to 25, and the most matches with a duration of 18 minutes')

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
        st.title('Hypothesis testing :eyes:')
        st.markdown(''' ### $H_0 : p_{radiant-win} = p_{dire-win} = 0.5$ ''')
        st.markdown(''' ### $H_A : p_{radiant-win} ≠ p_{dire-win}$ ''')
        st.caption('where p - probability')
        st.dataframe(return_count_victories(), use_container_width=True)
        #1.1
        st.markdown('## First approach: $Chi^2$')
        st.markdown('### Represent our dataset in the a different way')
        st.dataframe(pd.DataFrame([[4108, 3254], [3681, 3681]], columns=['radiant', 'dire'], index=['observed wins', 'expected wins']), use_container_width=True)
        st.markdown(r'''Lets calculate Chi-squared distance by this formula:
        
$χ^2 = \displaystyle\sum_{i=1}^{n} \frac{(observed_i - expected_i)^2}{expected_i} =  \frac{(4108-3681)^2}{3681} + \frac{(3254-3681)^2}{3681} = 99.06$

After we can calculate the p-value by plotting the distance value on the **distribution graph $Chi^2$**.''')
        st.markdown('''**We have one degree of freedom, hence the critical $Chi^2$ value for p-value = 0.05 is 3.84**''')
        chi_distrib = Image.open('pictures/Chi-distrib.png')
        st.image(chi_distrib)
        st.markdown('''The resulting p-value = 2.4e-23 is so small that it cannot be displayed on the chart''')
        st.markdown('''### Conclusion:round_pushpin:: 
**Based on the p-value, we reject $H_0$ and we can say that the distribution of wins of the two sides is not uniform :arrow_right: and since 
the match data was collected randomly and independently of any influences, we can say that at least in patch 7.32d, the percentage of wins of the :green[Radiant] side is higher than :red[Dire].**''')

        #1.2
        st.markdown('## Second approach: Gaussian approximation')
        st.markdown('''Binomial distribution is our case.
        \n
Since our $n$ is large, we can approximate the binomial distribution with a Gaussian, and we can directly look up $z$-score in a 
$p$-value table for Gaussian distribution.
\n
If the probability for "radiant_win" is equal to the probability of "dire_win", they are both 0.5.
\n
$n = 7362$, we can safely use a Gaussian approximation and calculate the z-score.''')

        st.markdown(r'''
### $z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}} = \frac{0.558 - 0.5}{\sqrt{\frac{0.5(1-0.5)}{7362}}} = 9.95$

where $\hat{p}$ is our estimated probability of 'radiant_win' and $p_0 = 0.5$''')
        st.markdown('''**In the table of the Gaussian distribution, we will not find such limiting z-values, so we calculate the p-value using scipy**''')
        st.code('p_value = scipy.stats.norm.sf(abs(z)) \n#this snippet returned: 1.22e-23', language='python')
        st.markdown('''### Conclusion:round_pushpin::
**The p-value is much less than the threshold value of 0.05, and we can safely conclude that the probability of "radiant_win" is statistically significantly different from "dire_win".**''')

    # with tab4:
    #     st.title('About model')
