import pandas as pd

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

country_list_region = pd.read_csv("/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/country_list_region.csv")
encoded_df = pd.get_dummies(country_list_region['Intermediate Region Name'], prefix='Region')
print(encoded_df)