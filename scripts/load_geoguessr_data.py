import os
import PIL
from PIL import Image
import pandas as pd

def load_data(DATA_PATH: str):
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