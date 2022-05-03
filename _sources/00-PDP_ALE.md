---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3.8.12 ('base')
  language: python
  name: python3
---

# Partial Dependence Plot

## Main Concept

The partial dependence plot(PDP) shows the marginal effect one or two features have on the predicted outcome of a machine learning model {cite}`friedman2001greedy`.

In the probability theory, the marginal probability of random variable $X$ on $Y$ can be calculated by getting expectation on the joint distribution of random variables $X, Y$.

$$ f_X(x) = \Bbb{E}_Y\left[f_{X, Y}(x, y)\right] = \int_Y f_{X, Y}(x, y) f_Y(y) dy$$

Let's say the random variable that we concern is $X_s$, and the other variables are $X_c$, the partial dependence function is as follow:

$$f_{X_s}(x_S)=\Bbb{E}_{X_c}\left[f_{X_s, X_c}(x_s,x_c)\right]=\int f_{X_s, X_c}(x_s,x_c)f_{X_c}(x_c)dx_c$$

- $x_s$: the features that we care about
- $x_c$: other features
- $f$: probability function(or model)

## Implementation

The partial function $f_{X_s}(x_S)$ can be estimated by calculating averages in the training data, also known as Monte Carlo method:

$$f_{X_s}(x_S) = \dfrac{1}{n} \sum_{i=1}^n f_{X_s, X_c}(x_s,x_c^{(i)})$$

In Scikit-learn it will generate a grid by the `percentile`(defalut as 0.05 to 0.95) and `grid_resolution`, then replace the values as $x_s$ to compute the distirbution.

+++

## Example

```{code-cell} ipython3
import sys
from pathlib import Path

main_path = str(Path().absolute().parent)
sys.path.append(main_path)

import matplotlib.pyplot as plt
import pandas as pd
import hvplot.pandas

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
```

### Dataset: California Housing dataset

This dataset was obtained from the StatLib repository. https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

It was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

An household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surpinsingly large values for block groups with few households and many empty houses, such as vacation resorts.

- References: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

| Columns | Explaination |
| --- | --- |
| MedHouseVal | The median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).| 
| MedInc | median income in block group |
| HouseAge | median house age in block group |
| AveRooms | average number of rooms per household |
| AveBedrms | average number of bedrooms per household |
| Population | block group population |
| AveOccup | average number of household members |
| Latitude | block group latitude |
| Longitude | block group longitude |

```{code-cell} ipython3
house = fetch_california_housing()
df = pd.DataFrame(house['data'], columns=house['feature_names'])
df[house['target_names'][0]] = house['target'] - house['target'].mean()
df.hvplot.hist(y='MedHouseVal', bins=30, xlabel='Median House Value(Centralized, $100,000)', ylabel='Count', title='The Centralized Median House Value Distribution')
```

Prediction will be measured by $R^2$ Score, which means [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)

$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$, $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \epsilon_i^2$

```{code-cell} ipython3
# Select the hyperparameters
seed = 0
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.1, random_state=seed)
params = {
    'loss': 'squared_error',
    'learning_rate': 0.1,
    'random_state': seed,
    'l2_regularization': 1e-3
}

model = HistGradientBoostingRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'The mean squared error (MSE) on test set is {mean_squared_error(y_true=y_test, y_pred=y_pred):.4f}')
print(f'The R2 Score on test set is {r2_score(y_true=y_test, y_pred=y_pred):.4f}')
```

```{code-cell} ipython3
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']

fig, axes = plt.subplots(2, 4, figsize=(16, 4))
display = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features,
    kind="both",
    subsample=50,
    n_jobs=3,
    grid_resolution=20,
    random_state=0,
    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
    n_cols=4,
    ax=axes.flatten()
)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','Latitude', 'Longitude']

fig, axes = plt.subplots(2, 4, figsize=(16, 4))
display = PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    features,
    kind="both",
    subsample=50,
    n_jobs=3,
    grid_resolution=20,
    random_state=0,
    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
    n_cols=4,
    ax=axes.flatten()
)
plt.tight_layout()
plt.show()
```

# Advantages

- The computation of partial dependence plots is intuitive
- If the feature for which you computed the PDP is not correlated with the other features, then the PDPs perfectly represent how the feature influences the prediction on average. In the uncorrelated case, the interpretation is clear
- Partial dependence plots are easy to implement.
- The calculation for the partial dependence plots has a causal interpretation.

+++

# Accumulated Local Effects

+++



# References

```{bibliography}
:filter: docname in docnames
```

## Partial Dependence Plot

- https://christophm.github.io/interpretable-ml-book/pdp.html
- https://scikit-learn.org/stable/modules/partial_dependence.html
- https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py
- https://soohee410.github.io/iml_pdp

## Accumulated Local Effects

+++
