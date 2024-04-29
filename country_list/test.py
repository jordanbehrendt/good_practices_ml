import pandas as pd

country_list_region = pd.read_csv("./country_list/country_list_region.csv")

region_index_list = []

for i in range(0,23):
    for index, row in country_list_region.iterrows():
        if row["One Hot Region"][i*3 + 1] == '1':
            region_index_list.append(index)

print(region_index_list)