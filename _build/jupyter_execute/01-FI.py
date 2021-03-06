#!/usr/bin/env python
# coding: utf-8

# # Feature Interaction
# 
# Part of articles are from {cite:p}`molnar2022` Chapter [8.3 Feature Interaction](https://christophm.github.io/interpretable-ml-book/interaction.html).
# 
# ## Main Concept: Friedman’s H-statistic
# 
# {cite:p}`friedman2008predictive` A function $F(\mathbf{x})$ is said to exhibit an interaction between two of its variables $x_j$ and $x_k$ if the difference in the value of $F(x)$ as a result of changing the value of $x_j$ depends on the value of $x_k$.
# 
# $$\Bbb{E}_{\mathbf{x}} = \Big\lbrack \dfrac{\partial^2 F(\mathbf{x})}{\partial x_j \partial x_k} \Big\rbrack^2 > 0$$
# 
# If there is no interaction between these($x_j, x_k$) variables, we can decompose the function $F(x)$ as follows:
# 
# $$F(\mathbf{x})=f_{\setminus j}(\mathbf{x}_{\setminus j})+f_{\setminus k}(\mathbf{x}_{\setminus k})$$
# 
# - $\mathbf{x}_{\setminus i}$: all variables except $x_i$
# 
# If a given variable $x_j$ interacts with **none** of the other variables($\mathbf{x}_{\setminus j}$), then the function will be:
# 
# $$F(\mathbf{x})=f_j(x_j) + f_{\setminus j}(\mathbf{x}_{\setminus j})$$
# 
# In this case $F(x)$ is said to be **additive** in $x_j$.
# 
# Here, let us the properties of centered partial dependence function to study interaction effects in the predictive model. We can setimated the dependence of predictive models on low cardinality subsets of the variables from the data:
# 
# $$\hat{F}_s(\mathbf{x}_s) = \dfrac{1}{N} \sum_{i=1}^{N} F(\mathbf{x}_s, \mathbf{x}_{\setminus s}) $$
# 
# If two variables $x_j$ and $x_k$ do not interact, the partial dependence of $F(\mathbf{x})$ on $\mathbf{x}_s = (x_j , x_k)$ can be decomposed into the sum of the respective partial dependences on each variable separately:
# 
# $$F_{jk}(x_j,x_k)=F_j(x_j)+F_k(x_k)$$
# 
# - $F_{jk}(x_j,x_k)$: the 2-way partial dependence function of both features
# - $F_j(x_j), F_k(x_k)$: the partial dependence functions of the single feature
# 
# If the variable $x_j$ does not interact with any other variable, then the predict function $F(\mathbf{x})$ can be write as:
# 
# $$F(\mathbf{x})=F_j(x_j)+F_{\setminus j}(x_{\setminus j})$$
# 
# - $F_{\setminus j}(x_{\setminus j})$: the partial dependence without feature $j$
# 
# To test for the presence of an interaction between two features $x_j, x_k$, the statistic will be as follows:
# 
# $$H^2_{jk} = \frac{\sum_{i=1}^N\left[ \hat{F}_{jk}(x_{j}^{(i)},x_k^{(i)})-\hat{F}_j(x_j^{(i)}) - \hat{F}_k(x_{k}^{(i)})\right]^2}{\sum_{i=1}^N {\hat{F}}^2_{jk}(x_j^{(i)},x_k^{(i)})}$$
# 
# It measures the fraction of variance of $\hat{F}_{jk}(x_j, x_k)$ without the variance of $j$ ($\hat{F}_j(x_j)$) and $k$ ($\hat{F}_k(x_k)$) over the
# data distribution $\sum_{i=1}^N {\hat{F}}^2_{jk}(x_j^{(i)},x_k^{(i)})$.
# 
# Similarly, a statistic for a specific variable $x_j$ interacts with any other variable:
# 
# $$H^2_{j}=\frac{\sum_{i=1}^N\left[F(x^{(i)})-\hat{F}_j(x_{j}^{(i)})-\hat{F}_{\setminus j}(x_{\setminus j}^{(i)})\right]^2}{\sum_{i=1}^N F^2(x^{(i)})}$$
# 
# It measures the fraction of variance of vairable $j$ over the data distribution.
# 
# - $H^2_{j} = 0$: no interaction at all
# - $H^2_{j} = 1$: all of the variance of the $PD_{jk}$ or $\hat{f}$ is explained by the sum of the partial dependence functions

# ## Example

# In[1]:


import sys
import hvplot.pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

main_path = str(Path().absolute().parent)
sys.path.append(main_path)

from src.CLhousing import load_model

ds, model = load_model(seed=0)


# we first see the partial dependence plots

# In[2]:


from itertools import combinations

features = ds.data.columns
two_way_features = list(combinations(features, 2))
fig, axes = plt.subplots(len(two_way_features) // 4, 4, figsize=(16, 16))

display = PartialDependenceDisplay.from_estimator(
    model,
    ds.X_train,
    two_way_features,
    n_jobs=3,
    grid_resolution=20,
    random_state=0,
    ax = axes.flatten()
)
plt.tight_layout()
plt.show()


# In[3]:


# https://github.com/scikit-learn/scikit-learn/issues/22383

features_idx = list(range(len(features)))
two_way_features_idx = list(combinations(features_idx, 2))

h_uni = {}
for i in features_idx:
    h_uni[i] = partial_dependence(
        estimator=model, 
        X=ds.X_train, 
        features=[i], 
        percentiles=(0.05, 0.95), 
        kind='average'
    )

h_bi = {}
for i, j in two_way_features_idx:
    h_bi[(i, j)] = partial_dependence(
        estimator=model, 
        X=ds.X_train, 
        features=[i, j], 
        percentiles=(0.05, 0.95), 
        kind='average'
    )

h = np.zeros((len(features_idx), len(features_idx)))

for i, j in two_way_features_idx:
    a = (h_bi[(i, j)]['average'] - h_uni[i]['average'].reshape(1, -1, 1) - \
            h_uni[j]['average'].reshape(1, 1, -1))**2
    b = (h_bi[(i, j)]['average'])**2
    h[i, j] = a.sum() / b.sum()


# In[4]:


fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.heatmap(
    data=pd.DataFrame(h, index=features, columns=features), 
    cmap='coolwarm', annot=True, fmt='.2g', linewidths=0.5, ax=ax
)
ax.set_title('H-statistics', fontsize=20)
plt.show()


# In[5]:


fig, axes = plt.subplots(len(two_way_features_idx) // 4, 4, figsize=(16, 18))
for k, ((i, j), ax) in enumerate(zip(two_way_features_idx, axes.flatten())):
    interaction = h_bi[(i, j)]['average'] - h_uni[i]['average'].reshape(1, -1, 1) -\
        h_uni[j]['average'].reshape(1, 1, -1)
    
    df_interaction = pd.DataFrame(interaction.squeeze(), 
        index=h_bi[(i, j)]['values'][0].round(2), 
        columns=h_bi[(i, j)]['values'][1].round(2)
    )
    sns.heatmap(df_interaction, cmap='coolwarm', ax=ax)
    ax.set_xlabel(two_way_features[k][0])
    ax.set_ylabel(two_way_features[k][1])

plt.tight_layout()
plt.show()


# # References
# 
# ```{bibliography}
# :filter: docname in docnames
# ```
