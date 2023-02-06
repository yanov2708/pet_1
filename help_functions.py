import requests, json, pickle
import pandas as pd


def return_heroes():
    url_heroes = 'https://api.opendota.com/api/heroes'
    response_heroes = requests.get(url_heroes)
    try:
        data_heroes = json.loads(response_heroes.content.decode("utf-8"))
        print('Amount dota heroes:', len(data_heroes))
        dict_hero_id = {data_hero['localized_name']: data_hero['id'] for data_hero in data_heroes}
    except:
        print('No json data_heroes returned', '\n')
    heroes_list = dict_hero_id.keys()
    return heroes_list, dict_hero_id


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

def load_df():
    df = pd.read_csv('df/matches7.csv')
    df['start_date2'] = pd.to_datetime(df['start_date'], unit='s')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df[['duration', 'radiant_score', 'dire_score', 'radiant_win', 'start_date2']]