# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:44:00 2021

@author: Jérémie Aucher
"""
### Declaration ###
import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
import base64
import shap
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


# Initialisation ###
loanColumn = 'SK_ID_CURR'
target = 'TARGET'
colW=350
colH=500
# url = "https://ja-p7-api.herokuapp.com/"
url = "http://127.0.0.1:5000/"

### For API Asking ###
def convToB64(data):
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def restoreFromB64Str(data_b64_str):
    return pickle.loads(base64.b64decode(data_b64_str.encode()))

def askAPI(apiName, url=url, params=None):
    url=url+str(apiName)
    resp = requests.post(url=url,params=params).text
    return restoreFromB64Str(resp)

@st.cache(suppress_st_warning=True)
def apiModelPrediction(data,loanNumber,columnName='SK_ID_CURR',url=url, modelName='lightgbm'):
    # Reccupération de l'index
    idx = getTheIDX(data,loanNumber,columnName)
    
    # Création du df d'une seule ligne contenant les infos du client
    data = data.iloc[[idx]]
    
    # Création des données à passer en arguments au format dictionnaire
    params = dict(data_b64_str=convToB64(data))
    
    # Interrogation de l'API et récupération des données au format dictionnaire
    dictResp = askAPI(apiName=modelName ,url=url, params=params)
    
    return dictResp['predExact'], dictResp['predProba']

### Load Data and More ###
@st.cache(suppress_st_warning=True)
def loadData():
    return pickle.load(open(os.getcwd()+'/pickle/dataRef.pkl', 'rb')),\
        pickle.load(open(os.getcwd()+'/pickle/dataCustomer.pkl', 'rb'))

@st.cache(suppress_st_warning=True)
def loadModel(modelName='model'):
    return askAPI(apiName=modelName)

@st.cache(suppress_st_warning=True)
def loadThreshold():
    return askAPI(apiName='threshold')

### Get Data ###
@st.cache(suppress_st_warning=True)
def getDFLocalFeaturesImportance(model,X,loanNumber,nbFeatures=12,inv=False):
    idx = getTheIDX(data=X,columnName=loanColumn,value=loanNumber)
    shap_values = shap.TreeExplainer(model).shap_values(X.iloc[[idx]])[0]
    
    if inv:
        shap_values *= -1
    
    dfShap = pd.DataFrame(shap_values, columns=X.columns.values)
    serieSignPositive = dfShap.iloc[0,:].apply(lambda col: True if col>=0 else False)

    serieValues = dfShap.iloc[0,:]
    serieAbsValues = abs(serieValues)
    return pd.DataFrame(
        {
            'values':serieValues,
            'absValues':serieAbsValues,
            'positive':serieSignPositive,
            'color':map(lambda x: 'red' if x else 'blue', serieSignPositive)
            }
        ).sort_values(
            by='absValues',
            ascending=False
            ).iloc[:nbFeatures,:].drop('absValues', axis=1)

def getTheIDX(data,value,columnName='SK_ID_CURR'):
    '''
    Retourne l'index correspondant à la 1ère valeur contenue dans value
    contenue dans la colonne columnName du Dataframe data.
    ''' 
    return data[data[columnName] == value].index[0]

def getGender(data, idx):
    gender = data[data.index == idx].iloc[0]['CODE_GENDER']
    if gender == 0:
        return 'Female'
    else:
        return 'Male'

@st.cache(suppress_st_warning=True)
def get_df_global_shap_importance(model, X):
    # Explain model predictions using shap library:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    return pd.DataFrame(
        zip(
            X.columns[np.argsort(np.abs(shap_values).mean(0))][::-1],
            np.sort(np.abs(shap_values).mean(0))[::-1]
        ),
        columns=['feature','importance']
    )

### Plot Chart ###
def gauge_chart(score, threshold):
    color="RebeccaPurple"
    if score<threshold:
        color="darkred"
    else:
        color="green"
    fig = go.Figure(
        go.Indicator(
            domain = {
                'x': [0, 0.9],
                'y': [0, 0.9]
                },
            value = score,
            mode = "gauge+number+delta",
            title = {
                'text': "Score"
                },
            gauge = {
                'axis':{
                    'range':[None, 1]
                    },
                'bar': {
                    'color': color
                    },
                'steps' : [
                 {
                     'range': [0, 0.1],
                     'color': "#ff0000"
                     },
                 {
                     'range': [0.1, 0.2],
                     'color': "#ff4d00"
                     },
                 {
                     'range': [0.2, 0.3],
                     'color': "#ff7400"
                     },
                 {
                     'range': [0.3, 0.4],
                     'color': "#ff9a00"
                     },
                 {
                     'range': [0.4, 0.5],
                     'color': "#ffc100"
                     },
                 {
                     'range': [0.5, 0.6],
                     'color': "#c5ff89"
                     },                 
                 {
                     'range': [0.6, 0.7],
                     'color': "#b4ff66"
                     },
                 {
                     'range': [0.7, 0.8],
                     'color': "#a3ff42"
                     },
                 {
                     'range': [0.8, 0.9],
                     'color': "#91ff1e"
                     },
                 {
                     'range': [0.9, 1],
                     'color': "#80f900"
                     }
                 ],
             'threshold' :{
                 'line':{
                     'color': color,
                     'width': 8
                     },
                 'thickness': 0.75,
                 'value': score
                 }
             },
            delta = {'reference': 0.5, 'increasing': {'color': "RebeccaPurple"}}
            ))
    return fig

@st.cache(suppress_st_warning=True)
def plotGlobalFeaturesImportance(model, X, nbFeatures=10):
    '''
    nbFeatures ---> (n_first_element)
    '''
    # Suppression de la colonne <target> si elle existe.
    X = X.drop(target, axis=1, errors='ignore')
    
    data = get_df_global_shap_importance(model, X)
    
    fig = go.Figure()
    y=data.head(nbFeatures)['importance']
    x=data.head(nbFeatures)['feature']
    fig.add_trace(go.Bar(x=x, y=y,
                         marker=dict(color=y,
                                     colorscale='viridis')))
    return fig

@st.cache(suppress_st_warning=True)
def plotLocalFeaturesImportance(model,X,loanNumber,nbFeatures=12):
    dfValuesSign = getDFLocalFeaturesImportance(
        model=model,
        X=X,
        loanNumber=loanNumber,
        nbFeatures=nbFeatures,
        inv=True
        )
    i = dfValuesSign.index
    fig = px.bar(dfValuesSign,
                 x='values',
                 y=i,
                 color='color',
                 orientation='h',
                 category_orders=dict(index=list(i)))
    fig.update_layout(
        yaxis={'title': None},
        xaxis={'title': None},
        showlegend=False
        )
    return fig

@st.cache(suppress_st_warning=True)
def plotDistOneFeature(dataRef,feature,valCust):
    '''
    Retourne une figure distplot basé sur la variable <feature> des données <data>.
    Affiche deux distribution en fonction de la valeur de la target.
    Affiche également une barre verticale représentant la valeur du client pour cette variable.
    '''
    
    x0 = dataRef[dataRef[target]==0][feature]
    x1 = dataRef[dataRef[target]==1][feature]
    del dataRef
    hist_data = [x0, x1]
    group_labels = ['Refusé', 'Accepté']
    fig = ff.create_distplot(hist_data, group_labels)
    fig.add_vline(x=valCust, line_width=3, line_dash="dash", line_color="red")
    return fig

@st.cache(suppress_st_warning=True)
def plotScatter2D(dataRef, listValCust):
    fig = px.scatter(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        color=target
        )
    fig.add_vline(x=listValCust[0][1], line_width=1, line_dash="solid", line_color="red")
    fig.add_hline(y=listValCust[1][1], line_width=1, line_dash="solid", line_color="red")
    
    fig.update_layout(showlegend=True)
    
    return fig

@st.cache(suppress_st_warning=True)
def plotScatter3D(dataRef, listValCust):
    fig = px.scatter_3d(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        z=listValCust[2][0],
        color=target
        )
    fig.update_layout(showlegend=False)
    
    fig.add_scatter3d(
        x=[listValCust[0][1]],
        y=[listValCust[1][1]],
        z=[listValCust[2][1]]
        )
    return fig