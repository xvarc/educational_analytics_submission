from src.data import get_data, clean_data
from src.build_features import impute_features,encoder
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def holdout(df):
    X = df.drop(columns='pass')
    y = df['pass']
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    return train_X, val_X, train_y, val_y

def model_tuning(train_X,train_y):
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                #    'min_samples_split': min_samples_split,
                #    'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    
    # Instanciate model
    model = RandomForestClassifier()

    # Instanciate Random Search
    search = RandomizedSearchCV(
        model, random_grid,
        n_jobs=-1, cv=5, n_iter=100, verbose=1,random_state = 0)

    search = search.fit(train_X,train_y)
    return search.best_estimator_

def train(train_X,train_y):
    # train optimised model
    model = RandomForestClassifier(n_estimators=1600,max_features='auto',max_depth=10,bootstrap=True)
    model.fit(train_X, train_y)
    return model
    # make a single prediction

if __name__ == '__main__':
    data = get_data()
    data = clean_data(data)
    data = impute_features(data)
    data = encoder(data)
    train_X, val_X, train_y, val_y = holdout(data)
    #data = model_tuning(train_X,train_y)
    model = train(train_X,train_y)
