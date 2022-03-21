from src.data import get_data, clean_data
from src.build_features import impute_features,encoder
from src.train_model import holdout,train
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def predict(model, row):
    result = model.predict(row)
    return result

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true)**2).mean())

def evaluate(model, X_test, y_test):
    
    X_test["y_pred"] = model.predict(X_test)
    
    #print(confusion_matrix(y_test, X_test["y_pred"]))
    #print(classification_report(y_test, X_test["y_pred"]))
    return accuracy_score(y_test, X_test["y_pred"])


def run_model():
    data = get_data()
    data = clean_data(data)
    data = impute_features(data)
    data = encoder(data)
    
    train_X, test_X, train_y, test_y = holdout(data)
    
    ## The below function runs a the RandomSearch that I initially used to find the best estimator. 
    #data = model_tuning(train_X,train_y)
    
    model = train(train_X,train_y)
    
    score = evaluate(model, test_X, test_y)
    print(f"Model accuracy score is {score}")
    
    # The below snippet can be used to input observations and predict whether the student will pass
    # I use train_X[:1] as an example here. 
    # observation = train_X[:1]
    # result = predict(model, observation)
    # print(result)

    
if __name__ == '__main__':
    run_model()
