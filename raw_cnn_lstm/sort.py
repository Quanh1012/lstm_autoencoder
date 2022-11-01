import os
import yaml
import pandas as pd

path = '/home/quanhhh/Documents/model/configs/best_results/'

dir_list = []
for file in os.listdir(path):
    if file.endswith('.yml'):
        dir_list.append(file)

df = pd.DataFrame(columns=['train_acc', 'val_acc', 'model'])

def Make_df(filename):
    with open(filename) as fh:
        dict_data = yaml.safe_load(fh)
    train_acc = dict_data['train_acc'][-1]
    val_acc = dict_data['val_acc'][-1]

    df.loc[len(df.index)] = [train_acc, val_acc, STR]


for STR in dir_list:
    Make_df(path + STR)

df_final = df.sort_values(by=['val_acc', 'train_acc'],ascending = False, ignore_index = True)
df_final = df_final.head(5)
print("Best 5 result: \n")
print(df_final)

print("\nBest Result: " + path + df_final.iloc[0,2])
