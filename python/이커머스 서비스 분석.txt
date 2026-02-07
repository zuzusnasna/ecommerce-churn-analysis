import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 
  
# metadata 
print(online_retail.metadata) 
  
# variable information 
print(online_retail.variables) 
df = pd.read_csv("Online Retail.csv")

df.head()
df.info()
df.describe()
df = df[df['CustomerID'].notna()].copy()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(str)
df = df[~df['InvoiceNo'].str.startswith('C')]
reference_date = df['InvoiceDate'].max()
reference_date
rfm = df.groupby('CustomerID').agg(
    last_purchase_date=('InvoiceDate', 'max'),
    frequency=('InvoiceNo', 'nunique'),
    monetary=('UnitPrice', lambda x: (x * df.loc[x.index, 'Quantity']).sum())
).reset_index()
rfm['recency'] = (reference_date - rfm['last_purchase_date']).dt.days
rfm['churn'] = (rfm['recency'] >= 60).astype(int)
final_df = rfm[['CustomerID', 'recency', 'frequency', 'monetary', 'churn']]
final_df.head()
final_df['churn'].value_counts(normalize=True)

final_df.groupby('churn')[['recency', 'frequency', 'monetary']].mean()
final_df['churn'].value_counts(normalize=True)

final_df.groupby('churn')[['recency', 'frequency', 'monetary']].mean()
X = final_df[['recency', 'frequency', 'monetary']]
y = final_df['churn']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=50,
    random_state=42
)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_tree))
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_model.coef_[0]
}).sort_values(by='coefficient', ascending=False)

coef_df
1. recency (+) -> 오래 안 산 고객일수록 이탈
2. frequency (-) -> 자주 산 고객일수록 유지
3. monetary (-) -> 돈 많이 쓴 고객일수록 유지
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': tree_model.feature_importances_
}).sort_values(by='importance', ascending=False)

importance_df
final_df['churn_prob'] = log_model.predict_proba(X)[:, 1]
fig = px.scatter_3d(final_df,
                    x = 'recency',
                    y = 'frequency',
                    z = 'monetary',
                    color='churn',
                    color_discrete_map={0 : 'blue', 1 : 'red'},
                    category_orders={'churn' : ['0', '1']},
                    opacity=0.6,
                    title='RFM distribution b Churn Status',
                    labels={'recency' : 'Recency (Days)',
                           'frequency' : 'Frequency',
                           'monetary' : 'Monetary ($)'})
fig.update_layout(margin=dict(l=0,r=0,b=0,t=50))
fig.show()
high_risk = final_df[
    (final_df['churn_prob'] > 0.7) &
    (final_df['monetary'] > final_df['monetary'].median())
]

high_risk.head()
