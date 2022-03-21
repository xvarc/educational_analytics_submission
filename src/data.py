import pandas as pd
import os


def get_data():
    path = f"{os.getcwd()}/raw_data/test_data.tsv"
    return pd.read_csv(path, sep='\t')

def clean_data(df):
    #remove duplicates 
    df.drop_duplicates(inplace=True)

    #drop unnecessary columns
    return df.drop(['notes','first','last'],axis=1)

if __name__ == '__main__':
    data = get_data()
    data = clean_data(data)
    #print(data)
