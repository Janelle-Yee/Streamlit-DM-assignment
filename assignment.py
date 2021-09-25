## Setup environment, load the relevant libraries

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from IPython.display import display
from streamlit_folium import folium_static
import folium
import csv

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(11,6)})

# Regression and Classification
import sklearn
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)
plt.rc("font", size=14)
st.set_option('deprecation.showPyplotGlobalUse', False)

#####################################################################################################################################################################
html_temp = """<div style="background-color:lightblue; padding:1.5px">
<h1 style="color:while; text-align: center;">TDS 3301 Data Mining Assignment</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

#st.title("TDS 3301 Data Mining Assignment")
st.markdown("Group members: Yee Wen San (1181101326), Fiona Liou Shin Wee (1181100812), Denis Siow Chin Hsuen (118110466)")
st.subheader(" ")
st.header("QUESTION 3: Python Programming")

df = pd.read_csv("df.csv")
df1 = pd.read_csv("df1.csv")
df_merge_state1 = pd.read_csv("df_merge_state1.csv")

######################################################################################################################################################################
#  Q3 (i) boxplot  
st.subheader(" ")
st.title("Q3 (i) Detection of outliers")
st.header("Boxplot for cases_state") 
plotbox = df[['cases_new', 'cases_import', 'cases_recovered']]
#plt.figure(figsize=(8, 8,))
bp = sns.boxplot(data = plotbox)
st.pyplot()

st.header("Boxplot for tests_state") 
plotbox = df1[['rtk-ag', 'pcr']]
#plt.figure(figsize=(8, 8,))
bp = sns.boxplot(data = plotbox)
st.pyplot()

#####################################################################################################################################################################
# Q3 (ii) correlation   
st.subheader(" ")
st.title("Q3 (ii) States that exhibit strong correlation with Pahang and Johor")
def display_correlation(df):
    r = df.corr(method="pearson")
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    st.header("Pearson Correlation")
    return(r)
q3ii = display_correlation(df_merge_state1)
st.pyplot()

#####################################################################################################################################################################
# Q3 (iii) BORUTA & RFE
st.subheader(" ")
st.title("Q3 (iii) Strong features to daily cases for Pahang, Kedah, Johor, and Selangor")

st.subheader(" ")
st.title("BORUTA")
st.header("(i) Pahang")
boruta_score=pd.read_csv("boruta_score_pahang.csv")
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("Boruta Top 10 Features for Pahang") 
st.pyplot()

st.header("(ii) Kedah")
boruta_score=pd.read_csv("boruta_score_kedah.csv")
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("Boruta Top 10 Features for Kedah") 
st.pyplot()

st.header("(iii) Johor")
boruta_score=pd.read_csv("boruta_score_johor.csv")
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("Boruta Top 10 Features for Johor") 
st.pyplot()

st.header("(iv) Selangor")
boruta_score=pd.read_csv("boruta_score_selangor.csv")
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("Boruta Top 10 Features for Selangor") 
st.pyplot()

####################

st.subheader(" ")
st.title("Recursive Feature Elimination (RFE)")
st.header("(i) Pahang")
rfe_score=pd.read_csv("rfe_score_pahang.csv")
sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("RFE Top 10 Features for Pahang") 
st.pyplot()

st.header("(ii) Kedah")
rfe_score=pd.read_csv("rfe_score_kedah.csv")
sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("RFE Top 10 Features for Kedah") 
st.pyplot()

st.header("(iii) Johor")
rfe_score=pd.read_csv("rfe_score_johor.csv")
sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("RFE Top 10 Features for Johor") 
st.pyplot()

st.header("(iv) Selangor")
rfe_score=pd.read_csv("rfe_score_selangor.csv")
sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:10], kind = "bar", height=5, aspect=2, palette='coolwarm')
st.subheader("RFE Top 10 Features for Selangor") 
st.pyplot()

#######################################################################################################################################################################
# Q3 (iv) Regression & Classification
filter1 = df[(df["date"]>= ('2021-01-01'))& (df["state"]=="Pahang")]
filter1 = filter1.drop(["date","state"], 1)

filter2 = df[(df["date"]>= ('2021-01-01'))& (df["state"]=="Kedah")]
filter2 = filter2.drop(["date","state"], 1)

filter3 = df[(df["date"]>= ('2021-01-01'))& (df["state"]=="Johor")]
filter3 = filter3.drop(["date","state"], 1)

filter4 = df[(df["date"]>= ('2021-01-01'))& (df["state"]=="Selangor")]
filter4 = filter4.drop(["date","state"], 1)

# Prepare dataset X and y.
X = filter1.drop("cases_new", 1)
y = filter1["cases_new"]

X1 = filter2.drop("cases_new", 1)
y1 = filter2["cases_new"]

X2 = filter3.drop("cases_new", 1)
y2 = filter3["cases_new"]

X3 = filter4.drop("cases_new", 1)
y3 = filter4["cases_new"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=2)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=2)


st.subheader(" ")
st.title("Q3 (iv) Comparing regression and classification models in predicting the daily cases for Pahang, Kedah, Johor, and Selangor")

st.header(" ")
st.title("(i) Pahang")
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Linear Regression")
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    p1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    p1

with right_column:
    st.subheader("Lasso Regression")
    lassoM = linear_model.Lasso(alpha=1.0,normalize=True, max_iter=1e5)
    lassoM.fit(X_train, y_train)
    y_pred = lassoM.predict(X_test)
    p2 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
    p2

plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
plt.scatter(y_test,y_pred)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='blue')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
st.subheader(" ")
plt.title("Regression plot: Actual vs Predicted")
st.pyplot()

left_column, right_column = st.columns(2)
with left_column:
    st.subheader(" ")
    st.subheader("Decision Tree Classifier")
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    classifier.score(X_test, y_test)  

    result_p = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred}) 
    result_p

with right_column:
    st.subheader(" ")
    st.subheader("K-Nearest Neighbors (KNN) Classifier")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)

    result_p1 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred}) 
    result_p1

#################
st.header(" ")
st.title("(ii) Kedah")
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Linear Regression")
    regressor = LinearRegression()
    regressor.fit(X1_train, y1_train)
    y1_pred = regressor.predict(X1_test)
    k1 = pd.DataFrame({'Actual': y1_test, 'Predicted': y1_pred})
    k1

with right_column:
    st.subheader("Lasso Regression")
    lassoM = linear_model.Lasso(alpha=1.0,normalize=True, max_iter=1e5)
    lassoM.fit(X1_train, y1_train)
    y1_pred = lassoM.predict(X1_test)
    k2 = pd.DataFrame({'Actual':y1_test, 'Predicted':y1_pred})  
    k2

plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
plt.scatter(y1_test,y1_pred)
plt.plot([np.min(y1_test), np.max(y1_test)], [np.min(y1_test), np.max(y1_test)], color='blue')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
st.subheader(" ")
plt.title("Regression plot: Actual vs Predicted")
st.pyplot()

left_column, right_column = st.columns(2)
with left_column:
    st.subheader(" ")
    st.subheader("Decision Tree Classifier")
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
    classifier.fit(X1_train, y1_train)
    y1_pred = classifier.predict(X1_test)
    classifier.score(X1_test, y1_test)  

    result_k1 = pd.DataFrame({'Actual':y1_test, 'Predicted':y1_pred}) 
    result_k1

with right_column:
    st.subheader(" ")
    st.subheader("K-Nearest Neighbors (KNN) Classifier")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X1_train, y1_train)
    knn.score(X1_test, y1_test)
    y1_pred = knn.predict(X1_test)

    result_k2 = pd.DataFrame({'Actual':y1_test, 'Predicted':y1_pred}) 
    result_k2

#######################
st.header(" ")
st.title("(iii) Johor")
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Linear Regression")
    regressor = LinearRegression()
    regressor.fit(X2_train, y2_train)
    y2_pred = regressor.predict(X2_test)
    j1 = pd.DataFrame({'Actual': y2_test, 'Predicted': y2_pred})
    j1

with right_column:
    st.subheader("Lasso Regression")
    lassoM = linear_model.Lasso(alpha=1.0,normalize=True, max_iter=1e5)
    lassoM.fit(X2_train, y2_train)
    y2_pred = lassoM.predict(X2_test)
    j2 = pd.DataFrame({'Actual':y2_test, 'Predicted':y2_pred})  
    j2

plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
plt.scatter(y2_test,y2_pred)
plt.plot([np.min(y2_test), np.max(y2_test)], [np.min(y2_test), np.max(y2_test)], color='blue')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
st.subheader(" ")
plt.title("Regression plot: Actual vs Predicted")
st.pyplot()


left_column, right_column = st.columns(2)
with left_column:
    st.subheader(" ")
    st.subheader("Decision Tree Classifier")
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
    classifier.fit(X2_train, y2_train)
    y2_pred = classifier.predict(X2_test)
    classifier.score(X2_test, y2_test)  

    result_j1 = pd.DataFrame({'Actual':y2_test, 'Predicted':y2_pred}) 
    result_j1

with right_column:
    st.subheader(" ")
    st.subheader("K-Nearest Neighbors (KNN) Classifier")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X2_train, y2_train)
    knn.score(X2_test, y2_test)
    y2_pred = knn.predict(X2_test)

    result_j2 = pd.DataFrame({'Actual':y2_test, 'Predicted':y2_pred}) 
    result_j2

######################
st.header(" ")
st.title("(iv) Selangor")
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Linear Regression")
    regressor = LinearRegression()
    regressor.fit(X3_train, y3_train)
    y3_pred = regressor.predict(X3_test)
    s1 = pd.DataFrame({'Actual': y3_test, 'Predicted': y3_pred})
    s1

with right_column:
    st.subheader("Lasso Regression")
    lassoM = linear_model.Lasso(alpha=1.0,normalize=True, max_iter=1e5)
    lassoM.fit(X3_train, y3_train)
    y3_pred = lassoM.predict(X3_test)
    s2 = pd.DataFrame({'Actual':y3_test, 'Predicted':y3_pred})  
    s2


plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
plt.scatter(y3_test,y3_pred)
plt.plot([np.min(y3_test), np.max(y3_test)], [np.min(y3_test), np.max(y3_test)], color='blue')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
st.subheader(" ")
plt.title("Regression plot: Actual vs Predicted")
st.pyplot()


left_column, right_column = st.columns(2)
with left_column:
    st.subheader(" ")
    st.subheader("Decision Tree Classifier")
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
    classifier.fit(X3_train, y3_train)
    classifier.score(X3_test, y3_test)
    y3_pred = classifier.predict(X3_test)
    
    result_s1 = pd.DataFrame({'Actual':y3_test, 'Predicted':y3_pred}) 
    result_s1

with right_column:
    st.subheader(" ")
    st.subheader("K-Nearest Neighbors (KNN) Classifier")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X3_train, y3_train)
    knn.score(X3_test, y3_test)
    y3_pred = knn.predict(X3_test)

    result_s2 = pd.DataFrame({'Actual':y3_test, 'Predicted':y3_pred}) 
    result_s2

st.title(" ")
st.info("-This is the end. Thank you.-")



