#!/usr/bin/env python
# coding: utf-8

# # Peer-to-Peer lending Domain
# 
# https://www.investopedia.com/terms/p/peer-to-peer-lending.asp
# 
# ## Tasks
# 
# - predict 
# 
# ## About Personal loan
# 
# A personal loan allows you to borrow money from a lender for almost any purpose, typically with a fixed term, a fixed interest rate, and a regular monthly payment schedule. Collateral usually is not required.
# 
# ## References
# 
# - https://www.lendingclub.com/investing/investor-education/interest-rates-and-fees
# - https://www.kaggle.com/code/aayush7kumar/lendingclub-loan-data-prediction
# 
# ## Peer-to-Peer Dataset
# 
# - https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data
# - https://www.kaggle.com/datasets/wordsforthewise/lending-club
# 
# ---

# # Lending Club
# 
# LendingClub is a peer-to-peer lending company headquartered in San Francisco California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC) and to offer loan trading on a secondary market. At its height LendingClub was the world's largest peer-to-peer lending platform. {cite:p}`wiki:LendingClub`
# 
# https://brunch.co.kr/@beyondplatform/4
# 
# ```{figure} ./figs/lendingclub.png
# ---
# height: 250px
# name: directive-fig
# ---
# lending club website
# ```
# 
# ## Business Understanding
# 
# The LendingClub company specialises in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:
# 
# - If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
# - If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company
# 
# The data given contains the information about past loan applicants and whether they 'defaulted' or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for takin actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.
# 
# When a person applies for a loan, there are two types of decisions that could be taken by the company:
# 
# 1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below:
#     - Fully paid: Applicant has fully paid the loan (the principal and the interest rate)
#     - Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.
#     - Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan
# 2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)
# 
# ## Business Metric
# 
# TBD
# 
# - It might be some explanation on why some of customer fails to pay the loan for reject purpose

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas
from pathlib import Path

main_path = Path().absolute().parent
data_path = main_path / 'data' / 'p2p' / 'lending_club' / 'processed'


# ## Exploratory Data Analysis
# 
# ### Data Description
# 
# |LoanStatNew|Description|
# |---|---|
# |loan_amnt|The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.|
# |term|The number of payments on the loan. Values are in months and can be either 36 or 60.|
# |loan_status|Current status of the loan.|
# |int_rate|Interest Rate on the loan.|
# |installment|The monthly payment owed by the borrower if the loan originates.|
# |grade|LC assigned loan grade.|
# |sub_grade|LC assigned loan subgrade.|
# |emp_title|The job title supplied by the Borrower when applying for the loan.|
# |emp_length|Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. |
# |home_ownership|The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
# |annual_inc|The self-reported annual income provided by the borrower during registration.|
# |verification_status|Indicates if income was verified by LC, not verified, or if the income source was verified.|
# |issue_d|The month which the loan was funded.|
# |purpose|A category provided by the borrower for the loan request. |
# |title|The loan title provided by the borrower.|
# |dti|A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowe's self-reported monthly income.|
# |earliest_cr_line|The month the borrower's earliest reported credit line was opened.|
# |open_acc|The number of open credit lines in the borrower's credit file.|
# |pub_rec|Number of derogatory public records.|
# |pub_rec_bankruptcies|Number of public record bankruptcies.|
# |revol_bal|Total credit revolving balance.|
# |revol_util|"Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.|
# |total_acc|The total number of credit lines currently in the borrower's credit file.|
# |initial_list_status|The initial listing status of the loan. Possible values are – W, F|
# |application_type|Indicates whether the loan is an individual application or a joint application with two co-borrowers.|
# |mort_acc|Number of mortgage accounts.|
# |addr_state|The state provided by the borrower in the loan application.|

# In[2]:


# import data
df = pd.read_csv( data_path / 'accepted.csv')
df.info()


# ### loan status
# 
# Current status of the loan.

# In[3]:


df['loan_status'].value_counts().hvplot.bar(
    title='Loan Status Counts', xlabel='Loan Status', ylabel='Count', 
    width=500, height=350, yformatter='%d'
)


# ### loan_amnt & installment
# 
# - loan_amnt: The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
# - installment: The monthly payment owed by the borrower if the loan originates.

# In[4]:


loan_amnt = df.hvplot.hist(
    y='loan_amnt', by='loan_status', subplots=False, 
    width=400, height=400, bins=50, alpha=0.4, title='Loan Amount', 
    xlabel='Loan Amount', ylabel='Counts', legend='top',
    yformatter='%d'
)
loan_amnt


# In[5]:


installment = df.hvplot.hist(
    y='installment', by='loan_status', subplots=False, 
    width=400, height=400, bins=50, alpha=0.4, title='Installment', 
    xlabel='Installment', ylabel='Counts', legend='top',
    yformatter='%d'
)
installment


# In[6]:


installment_box = df.hvplot.box(
    y='installment', subplots=True, by='loan_status', width=250, height=400, 
    title='Installment', xlabel='Loan Status', ylabel='Installment', legend=False
)

loan_amnt_box = df.hvplot.box(
    y='loan_amnt', subplots=True, by='loan_status', width=250, height=400, 
    title='Loan Amount', xlabel='Loan Status', ylabel='Loan Amount', legend=False
)

loan_amnt_box + installment_box


# ### term & int_rate
# 
# - term: The number of payments on the loan. Values are in months and can be either 36 or 60.
# - int_rate: Interest Rate on the loan.

# In[7]:


term = df.groupby(['loan_status'])[['term']].value_counts().rename('Count').hvplot.bar()
term.opts(
    title="Term", xlabel='Term / Loan Status', ylabel='Count',
    width=500, height=450, show_legend=True, yformatter='%d'
)


# In[8]:


int_rate = df.hvplot.hist(
    y='int_rate', by='loan_status', subplots=False, 
    width=400, height=400, bins=30, alpha=0.4, 
    title='Interest Rate', 
    xlabel='Interest Rate', ylabel='Counts', legend='top',
    yformatter='%d'
)
int_rate


# usually sort-term has lower interest rate

# In[9]:


df.loc[:, ['loan_status', 'term', 'int_rate']].hvplot.hist(
    y='int_rate', groupby='loan_status', by='term', subplots=False, 
    width=400, height=400, bins=30, alpha=0.4, 
    title='Interest Rate by term', xlabel='Interest Rate', ylabel='Counts', legend='top',
    yformatter='%d', dynamic=False
)


# Interesting relation between interest rate and installment it that can be calculate by the following formula if using "Equal repayment of principal and interest"

# In[10]:


def cal_amount_erpi(loan_amnt, int_rate, term):
    """
    loan_anmt: loan amount
    int_rate: interest rate, percentage
    term: in month
    """
    int_rate_monthly = int_rate / 100 / 12
    payment_monthly = loan_amnt * int_rate_monthly
    total_to_pay = payment_monthly * (1 + int_rate_monthly)**term
    return total_to_pay / ((1 + int_rate_monthly)**term - 1)


# In[11]:


df_temp = df.loc[:, ['installment', 'loan_amnt', 'int_rate', 'term']].copy()
df_temp['term'] = df_temp['term'].str.strip('months').str.strip().astype(np.int32)
df_temp['installment_cal'] = df_temp.apply(lambda x: cal_amount_erpi(x['loan_amnt'], x['int_rate'], x['term']), axis=1)


# not all the payment are following "Equal repayment of principal and interest"

# In[12]:


df_temp_diff = (df_temp['installment_cal'] - df_temp['installment'])
df_diff = df_temp_diff.agg(['mean', 'std'])
print(f'Difference of Mean: {df_diff["mean"]:.4f} Standard Deviation {df_diff["std"]:.4f}')
df_temp_diff.loc[abs(df_temp_diff) > 1].hvplot.hist(
    subplots=False, width=400, height=400, bins=50, alpha=0.4, 
    title='Differences between calculated installment (Diff > 1)', xlabel='Diff', ylabel='Counts', legend='top',
    yformatter='%d',
)


# ### grade & sub_grade
# 
# - grade: LC assigned loan grade.
# - sub_grade: LC assigned loan subgrade.

# In[13]:


grade = df.groupby(['loan_status'])[['grade']].value_counts().sort_index().hvplot.bar(
    width=400, height=400, title='Grade Distribution', xlabel='Grade', ylabel='Count', 
    legend='top', yformatter='%d'
)
grade


# In[14]:


sub_grade = df.groupby(['loan_status'])[['sub_grade']].value_counts().sort_index().hvplot.barh(
    width=400, height=800, title='Sub-Grade Distribution', xlabel='Sub-Grade', ylabel='Count', 
    legend='top', xformatter='%d'
)
sub_grade


# usually people don't charge off at grade A - B, usually C grade are more often charge off. but cannot say that the people who has lower grade pay less on their loan.

# In[15]:


df.loc[df['grade'].isin(['E', 'F', 'G'])].groupby(['loan_status', 'grade'])[['grade']].value_counts().rename('count')    .hvplot.table(title='Grade Count in E-G')


# ### home_ownership & purpose
# 
# - home_ownership: The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
# - purpose: A category provided by the borrower for the loan request.

# In[16]:


df.groupby(['loan_status'])[['home_ownership']].value_counts().rename('count').hvplot.table()


# In[17]:


home_ownership = df.groupby(['loan_status'])[['home_ownership']].value_counts().rename('Count').hvplot.bar()
home_ownership.opts(
    title='Home Ownership', xlabel='Home Ownership / Loan Status', ylabel='Count',
    width=700, height=450, show_legend=True, yformatter='%d'
)


# In[18]:


purpose = df.groupby(['loan_status'])[['purpose']].value_counts().rename('Count').hvplot.bar()
purpose.opts(
    title="Purpose", xlabel='Purpose / Loan Status', ylabel='Count',
    width=700, height=450, show_legend=True, yformatter='%d', xrotation=90
)


# ### annual_inc & verification_status
# 
# - annual_inc: The self-reported annual income provided by the borrower during registration.
# - verification_status: Indicates if income was verified by LC, not verified, or if the income source was verified

# In[19]:


df.groupby(['loan_status', 'verification_status'])['annual_inc'].describe().round(2).hvplot.table(
    title='Annual Income Table Description By Loan Status & Verification', height=200, width=700)


# In[20]:


(df.groupby(['loan_status'])[['verification_status']].value_counts() / df.groupby(['loan_status'])['verification_status'].count())    .rename('percentage').hvplot.table(title='Income Verified Rate', height=200)


# In[21]:


def is_outlier(x): 
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    upper = np.percentile(x, 75) + (iqr * 1.5)
    lower = np.percentile(x, 25) - (iqr * 1.5)

    return (x > upper) | (x < lower)


annual_inc = df.loc[~df.groupby(['loan_status', 'verification_status'])['annual_inc'].apply(is_outlier), 
    ['loan_status', 'verification_status', 'annual_inc']].hvplot.hist(
    y='annual_inc', by='loan_status', groupby='verification_status', subplots=False, 
    width=700, height=400, bins=40, alpha=0.4, title='Annual Income(1Q~3Q +/- 1.5*IQR) Distsribution', 
    xlabel='Annual Income', ylabel='Counts', legend='top', yformatter='%d', xformatter='%d', dynamic=False
)
annual_inc


# ### emp_title & emp_length
# 
# - emp_title: The job title supplied by the Borrower when applying for the loan.
# - emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.

# In[22]:


check_null = lambda x: x.isnull().sum()
df_emp_null = df.loc[:, ['emp_title', 'emp_length']].apply([check_null, pd.Series.nunique]).rename(index={'<lambda>': 'nnull'}).reset_index().hvplot.table(
    title='Job Title & Employment length: NA values', height=100
)
df_emp_top20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[:20].hvplot.table(
    title='Job Title Top 20'
)


# In[23]:


df_emp_null


# In[24]:


df_emp_top20


# In[25]:


df['emp_length'].fillna('unknown', inplace=True)
df['emp_title'].fillna('unknown', inplace=True)
df['emp_title'] = df['emp_title'].str.lower()  # Unify into lower cases


# In[26]:


df_emp_top20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[:20].hvplot.table(
    title='Job Title Top 20'
)
df_emp_top20


# In[27]:


df_emp_bottom20 = df['emp_title'].value_counts().reset_index().rename(columns={'index': 'emp_title', 'emp_title': 'count'})[-20:].hvplot.table(
    title='Job Title Bottom 20'
)
df_emp_bottom20


# In[28]:


print(df['emp_title'].nunique())


# titles are not normalized(or structured), too many unique titles in the data.

# In[29]:


from itertools import product

loan_status_order = ['Charged Off', 'Fully Paid']
emp_length_order = ['unknown', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
emp_length = df.groupby(['loan_status'])[['emp_length']].value_counts().reindex(list(product(*[loan_status_order, emp_length_order])))    .rename('Count').hvplot.barh(stacked=True, legend='right')
emp_length.opts(
    title='Employment Length in years', height=400, width=700, xlabel='Counts', ylabel='Employment Length in years', xformatter='%d'
)


# ### issue_d & earliest_cr_line
# 
# - issue_d: The month which the loan was funded.
# - earliest_cr_line: The month the borrower's earliest reported credit line was opened.
# 
# Most people try to do the loan near the 2016 and started to create their credit line at 2000

# In[30]:


df['issue_d'] = pd.to_datetime(df['issue_d'])
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])

fully_paid = df.loc[df['loan_status']=='Fully Paid', 'issue_d'].hvplot.hist(bins=35) 
charged_off = df.loc[df['loan_status']=='Charged Off', 'issue_d'].hvplot.hist(bins=35)

# fully_paid * charged_off
loan_issue_date = (fully_paid * charged_off).opts(
    title='Loan Issue Date Distribution', xlabel='Loan Issue Date', ylabel='Count',
    width=400, height=350, legend_cols=2, legend_position='top_right'
).opts(xrotation=45, yformatter='%d')

fully_paid = df.loc[df['loan_status']=='Fully Paid', 'earliest_cr_line'].hvplot.hist(bins=35) 
charged_off = df.loc[df['loan_status']=='Charged Off', 'earliest_cr_line'].hvplot.hist(bins=35)

earliest_cr_line = (fully_paid * charged_off).opts(
    title='Earliest reported credit line', xlabel='earliest_cr_line', ylabel='Count',
    width=400, height=350, legend_cols=2, legend_position='top_right'
).opts(xrotation=45, yformatter='%d')

loan_issue_date + earliest_cr_line


# ### title
# 
# title is similar to the purpose, will drop it later

# In[31]:


print(df['title'].isnull().sum())


# In[32]:


df['title'] = df['title'].str.lower()
df['title'].value_counts()[:10]


# ### dti, open_acc, total_acc
# 
# - dti: A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowe's self-reported monthly income.
# - open_acc: The number of open credit lines in the borrower's credit file.
# - total_acc: The total number of credit lines currently in the borrower's credit file.

# In[33]:


df['dti'].describe().reset_index().hvplot.table(title='DTI Table Description', height=250)
# Can DTI be 999?


# In[34]:


dti = df.hvplot.hist(
    y='dti', bins=50, width=400, height=350, 
    title="dti Distribution", xlabel='dti', ylabel='Count',
    yformatter='%d'
)
dti


# In[35]:


dti_sub = df.loc[df['dti'] <= 50].hvplot.hist(
    y='dti', by='loan_status', bins=50, width=400, height=350, subplots=False, 
    title="dti(<=50) Distribution", xlabel='dti', ylabel='Count', shared_axes=False,
    alpha=0.4, legend='top', yformatter='%d'
)

dti_sub2 = df.loc[df['dti'] > 50].hvplot.hist(
    y='dti', by='loan_status', bins=100, width=400, height=350, subplots=False, 
    title="dti(>50) Distribution", xlabel='dti', ylabel='Count', shared_axes=False,
    alpha=0.4, legend='top', yformatter='%d'
)

dti_sub + dti_sub2


# In[36]:


open_acc = df.hvplot.hist(
    y='open_acc', by='loan_status', bins=50, width=450, height=350, 
    title='The number of open credit lines', xlabel='The number of open credit lines', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d'
)

total_acc = df.hvplot.hist(
    y='total_acc', by='loan_status', bins=50, width=450, height=350, 
    title='The total number of credit lines', xlabel='The total number of credit lines', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d'
)

open_acc + total_acc


# ### revol_bal & revol_util
# 
# ```{admonition} What is revolving balance?
# 
# In credit card terms, a revolving balance is the portion of credit card spending that goes unpaid at the end of a billing cycle. 
# ```
# 
# https://www.capitalone.com/learn-grow/money-management/revolving-credit-balance/
# 
# - revol_bal: Total credit revolving balance.
# - revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

# In[37]:


df.groupby(['loan_status'])['revol_bal'].describe().round(2).reset_index().hvplot.table(title='Revolving Balance Table Description', height=100)


# In[38]:


revol_bal = df.hvplot.hist(
    y='revol_bal', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving Balance', xlabel='Revolving balance', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d', xformatter='%d'
).opts(xrotation=45)

revol_bal_sub = df.loc[df['revol_bal']<=250000].hvplot.hist(
    y='revol_bal', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving Balance(<=250000)', xlabel='Revolving balance', ylabel='Count', 
    alpha=0.4, legend='top', yformatter='%d', xformatter='%d', shared_axes=False
)
revol_bal + revol_bal_sub


# In[39]:


revol_util = df.hvplot.hist(
    y='revol_util', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving line utilization rate', xlabel='Revolving line utilization rate', ylabel='Count', 
    alpha=0.4, legend='top'
).opts(yformatter='%d')

revol_util_sub = df[df['revol_util'] < 120].hvplot.hist(
    y='revol_util', by='loan_status', bins=50, width=350, height=400, 
    title='Revolving line utilization rate (< 120)', xlabel='Revolving line utilization rate', ylabel='Count', 
    shared_axes=False, alpha=0.4, legend='top'
).opts(yformatter='%d')

revol_util + revol_util_sub


# ### pub_rec, pub_rec_bankruptcies & mort_acc
# 
# 
# ```{admonition} What is derogatory record?
# A derogatory public record is negative information on your credit report that is of a more serious nature and has become a matter of public record. It usually consists of bankruptcy filings, civil court judgments, foreclosures and tax liens. In some states, child support delinquencies are also a matter of public record.
# ```
# 
# https://budgeting.thenest.com/derogatory-public-record-mean-25266.html
# 
# 
# ```{admonition} What is a mortgage?
# The mortgage refers to a loan used to purchase or maintain a home, land, or other types of real estate. Usually paying the mortgage consistently will increse the credit score.
# ```
# 
# https://www.investopedia.com/terms/m/mortgage.asp
# https://www.investopedia.com/articles/personal-finance/031215/how-mortgages-affect-credit-scores.asp
# 
# - pub_rec: Number of derogatory public records.
# - pub_rec_bankruptcies: Number of public record bankruptcies.
# - mort_acc: Number of mortgage accounts.
# 
# From the data we can process these data as binary who had never have a public record versus more than once.

# In[40]:


pub_rec = df.groupby(['loan_status'])['pub_rec'].value_counts().rename('Count').hvplot.barh(
    title='The number of derogatory', xlabel='The number of derogatory', ylabel='Count',
    width=400, height=800, xformatter='%d'
)
pub_rec


# In[41]:


pub_rec_bankruptcies = df.groupby(['loan_status'])['pub_rec_bankruptcies'].value_counts().rename('Count').hvplot.barh(
    title='The number of public record bankruptcies', xlabel='The number of public record bankruptcies', ylabel='Count',
    width=400, height=600, xformatter='%d'
)
pub_rec_bankruptcies


# In[42]:


df['mort_acc'].describe().round(2)


# In[43]:


mort_acc = df.groupby(['loan_status'])['mort_acc'].value_counts().rename('Count').hvplot.barh(
    title='The number of mortgage accounts', xlabel='The number of mortgage accounts', ylabel='Count',
    width=400, height=700, xformatter='%d'
)

print(df['mort_acc'].isnull().sum())

mort_acc


# ### initial_list_status, application_type & addr_state
# 
# - initial_list_status: The initial listing status of the loan. Possible values are – W, F
# - application_type: Indicates whether the loan is an individual application or a joint application with two co-borrowers.
# - addr_state: The state provided by the borrower in the loan application.

# In[44]:


initial_list_status = df.groupby(['loan_status'])['initial_list_status'].value_counts().rename('Count').hvplot.bar(
    title='The initial listing status of the loan', xlabel='The initial listing status of the loan', ylabel='Count',
    width=400, height=400, yformatter='%d'
)
initial_list_status


# In[45]:


application_type = df.groupby(['loan_status'])['application_type'].value_counts().rename('Count').hvplot.bar(
    title='The application type', xlabel='The application type', ylabel='Count',
    width=400, height=400, yformatter='%d'
)
application_type


# In[46]:


addr_state = df.groupby(['loan_status'])['addr_state'].value_counts().rename('Count').hvplot.barh(
    title='The state provided by the borrower', xlabel='The state', ylabel='Count',
    width=500, height=850, xformatter='%d', legend='right'
)
addr_state


# ## Data Preprocessing
# 
# - Drop columns
# - Missing values
# - Detecting outlieres

# In[47]:


# reload the data
df = pd.read_csv( data_path / 'accepted.csv')
print(f'Data shape: {df.shape}')
print(df.columns)


# We will not use following columns: 
# - `title`: duplicated with purpose
# - `emp_title`: too many unique jobs, but seems like some of them are duplicated
# 
# Other columns 
# - `emp_length`: add 'unknown' for NaN values
# - `pub_rec`, `pub_rec_bankruptcies`: convert as binary
# - `verification_status`: combine verified together

# In[48]:


df.drop(columns=['title', 'emp_title'], inplace=True)


# check missing data

# In[49]:


for column in df.columns:
    missing_col = df[column].isnull().sum()
    if missing_col != 0:
        missing_percentage = (missing_col / len(df)) * 100
        print(f"'{column}': number of missing values {missing_col}({missing_percentage:.3f}%)")


# - 'emp_length' can add 'unknown' for NaN values
# - remove some outlier: dti, revol_util
# -

# ## Modeling

# ## Explanation on Models
