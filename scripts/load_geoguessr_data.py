import os
import PIL
from PIL import Image
import pandas as pd

def filter_min_img_df(df: pd.DataFrame, min_img: int):
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= min_img].index
    df = df[df['label'].isin(valid_labels)]
    return df

def load_data(DATA_PATH: str, min_img: int = 0, size_constraints: bool = False):
    list_rows = []
    for folder in os.listdir(DATA_PATH):
        for file in os.listdir(os.path.join(DATA_PATH,folder)):
            with PIL.Image.open(os.path.join(DATA_PATH,folder,file)) as img:
                width, height = img.size
                form = img.format
                temp_dict={
                    'label' : folder,
                    'width' : width,
                    'height' : height,
                    'format' : form,
                    'path' : os.path.join(DATA_PATH,folder,file)
                }
                list_rows.append(temp_dict)
    df = pd.DataFrame(list_rows)
    if size_constraints:
        df = df.loc[df['width'] == 1536]
    if min_img > 0:
        df = filter_min_img_df(df, min_img)
    return df