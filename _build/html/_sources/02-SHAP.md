---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3.10.4 ('XAI2022Spring-KSmjG5Tz')
  language: python
  name: python3
---

# SHAP

```{code-cell} ipython3
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import shap
import pandas as pd

shap.initjs()
```

# TreeSHAP explain

from https://medium.com/analytics-vidhya/shap-part-3-tree-shap-3af9bcd7cd9b

+++

## SHAP value calculation

```{code-cell} ipython3
from sklearn.tree import DecisionTreeRegressor, plot_tree

np.random.seed(100)
X_train = pd.DataFrame({
    'x':[206]*5 + [194] + [6]*4,
    'y': list(np.random.randint(100, 400, 6)) + [299, 299, 301, 301],
    'z': list(np.random.randint(100, 400, 10))
})
y_train = pd.Series([10]*5 + [20] + [50]*2 + [30]*2)
y_train.name = 't'
tree_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=2, min_samples_leaf=1, min_samples_split=2, random_state=100)
tree_model.fit(X=X_train, y=y_train)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
annots = plot_tree(tree_model, filled=True, ax=ax)
plt.show()
```

```{code-cell} ipython3
X_test = pd.DataFrame({'x': [150], 'y': [75], 'z': [200]})
```

```{code-cell} ipython3
import itertools

def iteration(iterable):
    s = list(iterable)
    return [list(itertools.permutations(s, r))[0] for r in range(1, len(s)+1)]

possible_players = list(itertools.permutations(['x', 'y', 'z']))

value_null = y_train.mean()
# f(x) given x = 150
value_x = 20.0
# f(y) given y = 75
value_y = 4/10 * 50 + 6/10 * ( 1/6*20 + 5/6*10 )
# f(z) given z = 200
value_z = value_null
vs = {'null': value_null, 'x': value_x, 'y': value_y, 'z': value_z}

all_phis = {}
all_preds = {}
all_preds[()] = value_null
for pc in possible_players:
    phis = []
    print(f'{pc}')
    for ps in iteration(pc):
        # prediction
        if 'x' in ps:
            pred = vs['x']
        elif 'y' in ps:
            pred = vs['y']
        else:
            pred = vs['z']
        
        if len(ps) == 1:
            phi = pred - vs['null']
            print(f'phi = {ps} - null = {pred} - {vs["null"]} = {phi}')
        elif len(ps) == 2:
            phi = pred - pred_prev
            print(f'phi = {ps} - {iteration(pc)[0]} = {pred} - {pred_prev} = {phi}')
        else:
            phi = pred - pred_prev
            print(f'phi = {ps} - {iteration(pc)[1]} = {pred} - {pred_prev} = {phi}')
        
        all_preds[tuple(sorted(ps))] = pred
        phis.append(float(phi))
        pred_prev = pred
    all_phis[pc] = phis
```

```{code-cell} ipython3
data = {'x': [], 'y': [], 'z': []}
for ks, vs in all_phis.items():
    for k, v in zip(*(ks, vs)):
        data[k].append(v)

df_all_phis = pd.DataFrame(data)
df_all_phis
```

```{code-cell} ipython3
# shape values are the average effect
shap_values = df_all_phis.mean(0).values
shap_values
```

```{code-cell} ipython3
explainer = shap.TreeExplainer(tree_model)
shap_values2 = explainer.shap_values(X_test)
shap_values2
```

```{code-cell} ipython3
shap.force_plot(explainer.expected_value, shap_values2[0,:], X_test.iloc[0,:])
```

## Feature Interaction in SHAP

https://christophm.github.io/interpretable-ml-book/shap.html#shap-interaction-values

https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html?highlight=pred_interactions%20#using-native-interface

$$\phi_{i,j}=\sum_{S\subseteq M \setminus\{i,j\}}\frac{\vert S\vert !(\vert M\vert -\vert S\vert -2)!}{2(\vert M\vert -1)!}\delta_{ij}(S)$$

where $i\neq j$ and 

$$\delta_{ij}(S)=\hat{f}_x(S\cup\{i,j\})-\hat{f}_x(S\cup\{i\})-\hat{f}_x(S\cup\{j\})+\hat{f}_x(S)$$

* $M$: total features set

```{code-cell} ipython3
def powerset(iterable):
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

def set_minus(x, y):
    if not isinstance(x, set):
        x = set(sorted(x))
    if not isinstance(y, set):
        y = set(sorted(y))
    return tuple(x - y)   

def set_plus(x, y):
    if not isinstance(x, set):
        x = set(sorted(x))
    if not isinstance(y, set):
        y = set(sorted(y))
    return tuple(sorted(x.union(y)))   

def interaction_value(all_preds, cols, interactions):
    phi = 0
    fact = sp.special.factorial
    M = len(cols)
    print(f'{interactions}')
    for s in powerset(set_minus(cols, interactions)):
        S = len(s)
        prob = (fact(S) * fact(M-S-2)) / (2 * fact(M-1))
        s_with_ij = set_plus(s, interactions)
        s_with_i = set_plus(s, interactions[0])
        s_with_j = set_plus(s, interactions[1])
        s_ij = (all_preds[s_with_ij] - all_preds[s_with_i] - all_preds[s_with_j] + all_preds[s])
        phi += s_ij * prob
        print(f'>> prob: {prob}')
        print(f'>> s_ij: f({s_with_ij}) - f({s_with_i}) - f({s_with_j}) + f({s}) = {s_ij}')
    return phi
```

```{code-cell} ipython3
cols = ['x', 'y', 'z']
all_preds
```

```{code-cell} ipython3
interaction_values = np.zeros([len(cols)]*2)
for interactions, idxes in zip(itertools.combinations(cols, 2), itertools.combinations(range(len(cols)), 2)):
    r, c = idxes
    iv = interaction_value(all_preds, cols, interactions)
    interaction_values[r, c] = iv
    interaction_values[c, r] = iv

r, c = np.tril_indices(len(cols), -1)
interactions_total = interaction_values[r, c].sum()
for i in shap_values.nonzero():  # to fit the missingness
    interaction_values[i, i] = shap_values[i] - interactions_total
```

```{code-cell} ipython3
# diag = main effect
# non-diag = interaction values
interaction_values
```

```{code-cell} ipython3
shap.TreeExplainer(tree_model).shap_interaction_values(X_test)
```
