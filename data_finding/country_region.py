import pandas as pd
import torch

""""
country_list = pd.read_csv("/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/country_list.csv")
region_list = pd.read_csv("/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/UNSD_Methodology.csv")

new_country_list = pd.merge(country_list, region_list, left_on="Alpha2Code", right_on="ISO-alpha2 Code")
new_country_list = new_country_list[["Country", "Alpha2Code", "Intermediate Region Name"]]
cl = country_list["Country"].tolist()
ncl = new_country_list["Country"].tolist()
unique_elements = list(set(cl) ^ set(ncl))
print(new_country_list)
new_country_list.to_csv("/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/country_list_region.csv", index=False)
"""

country_list_region = pd.read_csv("/home/leon/Documents/GPML/good_practices_ml/data_finding/country_list_region.csv")
encoded_df = pd.get_dummies(country_list_region['Intermediate Region Name'], prefix='Region')
one_hot_tensor = torch.tensor(encoded_df[encoded_df.columns].values, dtype=torch.float32)

# Convert the one-hot encoded DataFrame to a list of lists
one_hot_list = encoded_df.values.tolist()
one_hot_list = [[int(value) for value in row] for row in one_hot_list]

# Add the list of one-hot encoded vectors to the original DataFrame as a new column
country_list_region['One Hot Region'] = one_hot_list

encoded_df = pd.get_dummies(country_list_region['Country'])
one_hot_tensor = torch.tensor(encoded_df[encoded_df.columns].values, dtype=torch.float32)

# Convert the one-hot encoded DataFrame to a list of lists
one_hot_list = encoded_df.values.tolist()
one_hot_list = [[int(value) for value in row] for row in one_hot_list]

# Add the list of one-hot encoded vectors to the original DataFrame as a new column
country_list_region['One Hot Country'] = one_hot_list
country_list_region.to_csv("/home/leon/Documents/GPML/good_practices_ml/data_finding/country_list_region.csv", index=False)