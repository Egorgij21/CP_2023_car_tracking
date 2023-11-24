from glob import glob

import pandas as pd

paths = glob("PATH/*.csv")

save_to = 'PATH_TO_SAVE.csv'

root = paths[0].split('/')[:-1]

df = pd.concat( 
    map(pd.read_csv, paths), ignore_index=True)

df.to_csv(save_to, index=False)
