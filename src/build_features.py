from sklearn.impute import SimpleImputer
from src.data import get_data,clean_data
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder


def impute_features(df):
    #feature imputation
    #df.hours_studied.unique() # Check unique values in column
    hours_imputer = SimpleImputer(strategy="mean") 
    hours_imputer.fit(df[['hours_studied']]) # Fit imputer to hours column
    df['hours_studied'] = hours_imputer.transform(df[['hours_studied']]) # Impute
    return df

def scaler(df):
    rb_scaler = RobustScaler() 
    df['age'],df['hours_studied'] = rb_scaler.fit_transform(df[['age','hours_studied']]).T
    return df

def encoder(df):
    ohe = OneHotEncoder(sparse=False)
    enginetype_ohe = ohe.fit_transform(df[['country', 'lang','sex']])
    df['Australia'],df['Finland'],df['France'],df['Italy'],df['Japan'],df['Mexico'],df['New Zealand'],df['Spain'],df['UK'],df['USA'],df['English'],df['Finnish'],df['French'],df['Italian'],df['Japanese'],df['Spanish'],df['Female'], df['Male'] = enginetype_ohe.T
    df.drop(columns=['country','lang', 'sex'], inplace = True)
    return df

if __name__ == '__main__':
    data = get_data()
    data = clean_data(data)
    data = impute_features(data)
    data = encoder(data)
    print(data)
