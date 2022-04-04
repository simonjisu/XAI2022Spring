---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Peer-to-Peer lending Domain

https://www.investopedia.com/terms/p/peer-to-peer-lending.asp

## Tasks

- predict 

## About Personal loan

A personal loan allows you to borrow money from a lender for almost any purpose, typically with a fixed term, a fixed interest rate, and a regular monthly payment schedule. Collateral usually is not required.

## References

- https://www.lendingclub.com/investing/investor-education/interest-rates-and-fees
- https://www.kaggle.com/code/aayush7kumar/lendingclub-loan-data-prediction

## Peer-to-Peer Dataset

- https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data
- https://www.kaggle.com/datasets/wordsforthewise/lending-club

---

+++

# Lending Club

LendingClub is a peer-to-peer lending company headquartered in San Francisco California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC) and to offer loan trading on a secondary market. At its height LendingClub was the world's largest peer-to-peer lending platform. {cite}`wiki:LendingClub`


```{figure} ./figs/lendingclub.png
---
height: 250px
name: directive-fig
---
lending club website
```

## Business Understanding

The LendingClub company specialises in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:

- If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
- If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company

The data given contains the information about past loan applicants and whether they 'defaulted' or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for takin actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.

When a person applies for a loan, there are two types of decisions that could be taken by the company:

1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below:
    - Fully paid: Applicant has fully paid the loan (the principal and the interest rate)
    - Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.
    - Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan
2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)


```{code-cell} ipython3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas
from pathlib import Path

main_path = Path().absolute().parent
```

## Exploratory Data Analysis

### Data Description

|LoanStatNew|Description|
|---|---|
|loan_amnt|The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.|
|term|The number of payments on the loan. Values are in months and can be either 36 or 60.|
|loan_status|Current status of the loan.|
|int_rate|Interest Rate on the loan.|
|installment|The monthly payment owed by the borrower if the loan originates.|
|grade|LC assigned loan grade.|
|sub_grade|LC assigned loan subgrade.|
|emp_title|The job title supplied by the Borrower when applying for the loan.|
|emp_length|Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. |
|home_ownership|The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
|annual_inc|The self-reported annual income provided by the borrower during registration.|
|verification_status|Indicates if income was verified by LC, not verified, or if the income source was verified.|
|issue_d|The month which the loan was funded.|
|purpose|A category provided by the borrower for the loan request. |
|title|The loan title provided by the borrower.|
|dti|A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowe's self-reported monthly income.|
|earliest_cr_line|The month the borrower's earliest reported credit line was opened.|
|open_acc|The number of open credit lines in the borrower's credit file.|
|pub_rec|Number of derogatory public records.|
|pub_rec_bankruptcies|Number of public record bankruptcies.|
|revol_bal|Total credit revolving balance.|
|revol_util|"Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.|
|total_acc|The total number of credit lines currently in the borrower's credit file.|
|initial_list_status|The initial listing status of the loan. Possible values are – W, F|
|application_type|Indicates whether the loan is an individual application or a joint application with two co-borrowers.|
|mort_acc|Number of mortgage accounts.|
|addr_state|The state provided by the borrower in the loan application.|

```{code-cell} ipython3
data_path = main_path / 'data' / 'p2p' / 'lending_club' / 'processed'
# import data
df = pd.read_csv( data_path / 'accepted.csv')

df.info()
```

### loan status

Current status of the loan.

```{code-cell} ipython3
df['loan_status'].value_counts().hvplot.bar(
    title="Loan Status Counts", xlabel='Loan Status', ylabel='Count', 
    width=500, height=350, yformatter='%d'
)
```

### loan_amnt, installment & int_rate

- loan_amnt: The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
- installment: The monthly payment owed by the borrower if the loan originates.
- int_rate: Interest Rate on the loan.

```{code-cell} ipython3
:tags: [hide-input]

installment = df.hvplot.hist(
    y='installment', by='loan_status', subplots=False, 
    width=400, height=400, bins=50, alpha=0.4, 
    title="Installment by Loan Status", 
    xlabel='Installment', ylabel='Counts', legend='top',
    yformatter='%d'
)

loan_amnt = df.hvplot.hist(
    y='loan_amnt', by='loan_status', subplots=False, 
    width=400, height=400, bins=30, alpha=0.4, 
    title="Loan Amount by Loan Status", 
    xlabel='Loan Amount', ylabel='Counts', legend='top',
    yformatter='%d'
)

int_rate = df.hvplot.hist(
    y='int_rate', by='loan_status', subplots=False, 
    width=400, height=400, bins=30, alpha=0.4, 
    title='Interest Rate by Loan Status', 
    xlabel='Interest Rate', ylabel='Counts', legend='top',
    yformatter='%d'
)

installment + loan_amnt + int_rate
```

```{code-cell} ipython3
:tags: [hide-input]

installment_box = df.hvplot.box(
    y='installment', subplots=True, by='loan_status', width=300, height=400, 
    title='Loan Status by Installment', xlabel='Loan Status', ylabel='Installment', legend=False
)

loan_amnt_box = df.hvplot.box(
    y='loan_amnt', subplots=True, by='loan_status', width=300, height=400, 
    title='Loan Status by Loan Amount', xlabel='Loan Status', ylabel='Loan Amount', legend=False
)

int_rate_box = df.hvplot.box(
    y='int_rate', subplots=True, by='loan_status', width=300, height=400, 
    title='Loan Status by Interest Rate', xlabel='Loan Status', ylabel='Interest Rate', legend=False
)

loan_amnt_box + installment_box + int_rate_box
```

### grade & sub_grade

- grade: LC assigned loan grade.
- sub_grade: LC assigned loan subgrade.

```{code-cell} ipython3
:tags: [hide-input]

fully_paid = df.loc[df['loan_status']=='Fully Paid', 'grade'].value_counts().hvplot.bar() 
charged_off = df.loc[df['loan_status']=='Charged Off', 'grade'].value_counts().hvplot.bar() 

(fully_paid * charged_off).opts(
    title="Loan Status by Grade", xlabel='Grades', ylabel='Count',
    width=500, height=450, legend_cols=2, legend_position='top_right'
)
```

```{code-cell} ipython3
:tags: [hide-input]

fully_paid = df.loc[df['loan_status'] == 'Fully Paid', 'sub_grade'].value_counts().hvplot.bar() 
charged_off = df.loc[df['loan_status'] == 'Charged Off', 'sub_grade'].value_counts().hvplot.bar() 

(fully_paid * charged_off).opts(
    title="Loan Status by Grade", xlabel='Grades', ylabel='Count',
    width=500, height=400, legend_cols=2, legend_position='top_right', xrotation=90
)
```

usually giving the money to grade A - D, but cannot say that the people who has lower grade pay less on their loan. 

```{code-cell} ipython3
df.loc[df['grade'].isin(['E', 'F', 'G'])].groupby(['loan_status', 'grade'])['grade'].count()
```

### term, home_ownership,  & purpose

- term: The number of payments on the loan. Values are in months and can be either 36 or 60.
- home_ownership: The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
- purpose: A category provided by the borrower for the loan request.

```{code-cell} ipython3
df.groupby(['loan_status'])[['home_ownership']].value_counts().rename('Count').hvplot.table()
```

```{code-cell} ipython3
:tags: [hide-input]

home_ownership = df.groupby(['loan_status'])[['home_ownership']].value_counts().rename('Count').hvplot.bar()
home_ownership.opts(
    title="Home Ownership by Loan Status", xlabel='Home Ownership / Loan Status', ylabel='Count',
    width=900, height=450, show_legend=True, yformatter='%d'
)
```

```{code-cell} ipython3
:tags: [hide-input]

term = df.groupby(['loan_status'])[['term']].value_counts().rename('Count').hvplot.bar()
term.opts(
    title="Term by Loan Status", xlabel='Term / Loan Status', ylabel='Count',
    width=600, height=450, show_legend=True, yformatter='%d'
)
```

```{code-cell} ipython3
:tags: [hide-input]

term = df.groupby(['loan_status'])[['purpose']].value_counts().rename('Count').hvplot.bar()
term.opts(
    title="Purpose by Loan Status", xlabel='Purpose / Loan Status', ylabel='Count',
    width=700, height=450, show_legend=True, yformatter='%d', xrotation=90
)
```

### annual_inc & verification_status

- annual_inc: The self-reported annual income provided by the borrower during registration.
- verification_status: Indicates if income was verified by LC, not verified, or if the income source was verified

```{code-cell} ipython3
df.groupby(['loan_status', 'verification_status'])['annual_inc'].describe().hvplot.table(title='Annual Income Table Description By Verification')
```

```{code-cell} ipython3
(df.groupby(['loan_status'])[['verification_status']].value_counts() / df.groupby(['loan_status'])['verification_status'].count())\
    .rename('percentage').hvplot.table(title='Income Verified Rate')
```

```{code-cell} ipython3
:tags: [hide-input]

def is_outlier(x): 
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    upper = np.percentile(x, 75) + (iqr * 1.5)
    lower = np.percentile(x, 25) - (iqr * 1.5)

    return (x > upper) | (x < lower)


annual_inc = df.loc[~df.groupby(['loan_status', 'verification_status'])['annual_inc'].apply(is_outlier), 
    ['loan_status', 'verification_status', 'annual_inc']].hvplot.hist(
    y='annual_inc', by='loan_status', groupby='verification_status', subplots=False, 
    width=900, height=400, bins=40, alpha=0.4, title='Annual Income(1Q~3Q +/- 1.5*IQR) Distsribution by Loan Status', 
    xlabel='Annual Income', ylabel='Counts', legend='top', yformatter='%d', xformatter='%d', dynamic=False
)
annual_inc
```

### emp_title & emp_length

- emp_title: The job title supplied by the Borrower when applying for the loan.
- emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.

```{code-cell} ipython3
:tags: [hide-input]

check_null = lambda x: x.isnull().sum()
df_emp_null = df.loc[:, ['emp_title', 'emp_length']].apply([check_null, pd.Series.nunique]).rename(index={'<lambda>': 'nnull'}).reset_index().hvplot.table(
    title='Job Title & Employment length: NA values', height=100
)
df_emp_top20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[:20].hvplot.table(
    title='Job Title Top 20'
)
```

```{code-cell} ipython3
df_emp_null
```

```{code-cell} ipython3
df_emp_top20
```

```{code-cell} ipython3
df['emp_length'].fillna('unknown', inplace=True)
df['emp_title'].fillna('unknown', inplace=True)
df['emp_title'] = df['emp_title'].str.lower()  # Unify into lower cases
```

```{code-cell} ipython3
:tags: [hide-input]

df_emp_top20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[:20].hvplot.table(
    title='Job Title Top 20'
)
df_emp_top20
```

```{code-cell} ipython3
:tags: [hide-input]

from itertools import product

loan_status_order = ['Charged Off', 'Fully Paid']
emp_length_order = ['unknown', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_length = df.groupby(['loan_status'])[['emp_length']].value_counts().reindex(list(product(*[loan_status_order, emp_length_order]))).rename('Counts').hvplot.barh(stacked=True, legend='right')
emp_length.opts(
    title='Loan Status by Employment Length in years', height=400, width=900, xlabel='Counts', ylabel='Employment Length in years', xformatter='%d'
)
```

### issue_d & earliest_cr_line

- issue_d: The month which the loan was funded.
- earliest_cr_line: The month the borrower's earliest reported credit line was opened.

```{code-cell} ipython3
:tags: [hide-input]

df['issue_d'] = pd.to_datetime(df['issue_d'])
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

fully_paid = df.loc[df['loan_status']=='Fully Paid', 'issue_d'].hvplot.hist(bins=35) 
charged_off = df.loc[df['loan_status']=='Charged Off', 'issue_d'].hvplot.hist(bins=35)

# fully_paid * charged_off
loan_issue_date = (fully_paid * charged_off).opts(
    title="Loan Status by Loan Issue Date", xlabel='Loan Issue Date', ylabel='Count',
    width=450, height=350, legend_cols=2, legend_position='top_right'
).opts(xrotation=45, yformatter='%d')

fully_paid = df.loc[df['loan_status']=='Fully Paid', 'earliest_cr_line'].hvplot.hist(bins=35) 
charged_off = df.loc[df['loan_status']=='Charged Off', 'earliest_cr_line'].hvplot.hist(bins=35)

earliest_cr_line = (fully_paid * charged_off).opts(
    title="Loan Status by earliest_cr_line", xlabel='earliest_cr_line', ylabel='Count',
    width=450, height=350, legend_cols=2, legend_position='top_right'
).opts(xrotation=45, yformatter='%d')

loan_issue_date + earliest_cr_line
```

### title

```{code-cell} ipython3
print(df['title'].isnull().sum())
```

```{code-cell} ipython3
df['title'] = df['title'].str.lower()
df['title'].value_counts()[:10]
```


### dti, open_acc, pub_rec, pub_rec_bankruptcies

- dti: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowe's self-reported monthly income.
- open_acc: The number of open credit lines in the borrower's credit file.
- pub_rec: Number of derogatory public records.
- pub_rec_bankruptcies: Number of public record bankruptcies.

```{code-cell} ipython3
df['dti'].describe().reset_index().hvplot.table(title='DTI Table Description', height=250)
# Can DTI be 999?
```

```{code-cell} ipython3
:tags: [hide-input]

dti = df.hvplot.hist(
    y='dti', bins=50, width=450, height=350, 
    title="dti Distribution", xlabel='dti', ylabel='Count'
).opts(yformatter='%d')
dti_sub = df.loc[df['dti'] < 100].hvplot.hist(
    y='dti', bins=50, width=450, height=350, 
    title="dti(<100) Distribution", xlabel='dti', ylabel='Count', shared_axes=False
).opts(yformatter='%d')

dti_sub2 = df.loc[df['dti'] > 40].hvplot.hist(
    y='dti', bins=100, width=450, height=350, 
    title="dti(>40) Distribution", xlabel='dti', ylabel='Count', shared_axes=False
).opts(yformatter='%d')

dti + dti_sub + dti_sub2
```

```{code-cell} ipython3
:tags: [hide-input]

dti = df[df['dti']<=50].hvplot.hist(
    y='dti', by='loan_status', bins=50, width=450, height=350, 
    title="dti (<=50) Distribution", xlabel='dti', ylabel='Count', 
    alpha=0.3, legend='top'
).opts(yformatter='%d')

open_acc = df.hvplot.hist(
    y='open_acc', by='loan_status', bins=50, width=450, height=350, 
    title='Loan Status by The number of open credit lines', xlabel='The number of open credit lines', ylabel='Count', 
    alpha=0.4, legend='top'
).opts(yformatter='%d')

total_acc = df.hvplot.hist(
    y='total_acc', by='loan_status', bins=50, width=450, height=350, 
    title='Loan Status by The total number of credit lines', xlabel='The total number of credit lines', ylabel='Count', 
    alpha=0.4, legend='top'
).opts(yformatter='%d')

dti + open_acc + total_acc
```

### revol_bal & revol_util

```{admonition} What is Revolving balance?

In credit card terms, a revolving balance is the portion of credit card spending that goes unpaid at the end of a billing cycle. 
```

- revol_bal: Total credit revolving balance.
- revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

```{code-cell} ipython3
:tags: [hide-input]

revol_util = df.hvplot.hist(
    y='revol_util', by='loan_status', bins=50, width=450, height=400, 
    title='Loan Status by Revolving line utilization rate', xlabel='Revolving line utilization rate', ylabel='Count', 
    alpha=0.4, legend='top'
).opts(yformatter='%d')

revol_util_sub = df[df['revol_util'] < 120].hvplot.hist(
    y='revol_util', by='loan_status', bins=50, width=550, height=400, 
    title='Loan Status by Revolving line utilization rate (< 120)', xlabel='Revolving line utilization rate', ylabel='Count', 
    shared_axes=False, alpha=0.4, legend='top'
).opts(yformatter='%d')

revol_util + revol_util_sub
```
