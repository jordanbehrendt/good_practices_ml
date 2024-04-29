import pandas as pd
import numpy as np



country_list_region = pd.read_csv('./country_list/country_list_region.csv')
UNSD = pd.read_csv('./country_list/UNSD_Methodology.csv')

find_continent_name = lambda x: UNSD.loc[UNSD['Intermediate Region Name'] == x].iloc[0]['Region Name']
country_list_region['Continent'] = country_list_region['Intermediate Region Name'].apply(find_continent_name)

country_list_region.to_csv('./country_list/country_list_region_and_continent.csv')