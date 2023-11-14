import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import math

def log(x):
    log_value = math.log(x)
    if log_value == 0:
        log_value = 1
    return log_value

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
image_distribution = pd.read_csv("image_distribution.csv")
count_column = image_distribution["count"]
new_count_column = count_column.apply(log)
image_distribution["count"] = new_count_column

world['join'] = 1
image_distribution['join'] = 1

dataFrameFull = world.merge(image_distribution, on='join').drop('join', axis=1)
image_distribution.drop('join', axis=1, inplace=True)

dataFrameFull['match'] = dataFrameFull.apply(lambda x: x["name"].find(x["label"]), axis=1).ge(0)

df = dataFrameFull[dataFrameFull['match']]

df.plot(column='count', legend=True)
plt.savefig("world.jpg")