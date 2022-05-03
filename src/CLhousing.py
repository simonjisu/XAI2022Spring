import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class CLDataset:
    def __init__(self, seed):
        self.house = fetch_california_housing()
        self.df = pd.DataFrame(self.house['data'], columns=self.house['feature_names'])

        # preprocessing
        self.y_mean = self.house['target'].mean()
        self.df[self.house['target_names'][0]] = self.house['target'] - self.y_mean
        self.data = self.df.iloc[:, :-1]
        self.target = self.df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.1, random_state=seed)
        

def load_model(seed:int=0):
    ds = CLDataset(seed)
    
    params = {
        'loss': 'squared_error',
        'learning_rate': 0.1,
        'random_state': seed,
        'l2_regularization': 1e-3
    }

    model = HistGradientBoostingRegressor(**params)
    model.fit(ds.X_train, ds.y_train)

    y_pred = model.predict(ds.X_test)
    print(f'The mean squared error (MSE) on test set is {mean_squared_error(y_true=ds.y_test, y_pred=y_pred):.4f}')
    print(f'The R2 Score on test set is {r2_score(y_true=ds.y_test, y_pred=y_pred):.4f}')

    return ds, model