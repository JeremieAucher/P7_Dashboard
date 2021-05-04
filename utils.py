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
import plotly.figure_factory as ff
import plotly.graph_objects as go
import shap
import plotly.express as px

# Initialisation ###
loanColumn = 'SK_ID_CURR'
target = 'TARGET'
colW=350
colH=500

### Chart - Start ###
def gauge_chart(score, threshold):
    # score *= 100
    color="RebeccaPurple"
    if score<threshold:
        color="darkred"
    # elif score>= threshold*1.1:
        # color="orange"
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
    # fig.update_layout(
    #     width=750,
    #     height=500
    #     )
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
    # fig.update_layout(
    #     margin=dict(l=20, r=20, t=20, b=20),
    #     width=colW,
    #     height=colH,
    #     paper_bgcolor="LightSteelBlue"
    #     )
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
        showlegend=False,
        # margin=dict(l=20, r=20, t=20, b=20),
        # width=colW,
        # height=colH,
        # paper_bgcolor="LightSteelBlue"
        )
    return fig

@st.cache(suppress_st_warning=True)
def plotDistOneFeature(dataRef,feature,valCust):
    '''
    Retourne une figure distplot basé sur la variable <feature> des données <data>.
    Affiche deux distribution en fonction de la valeur de la target.
    Affiche également une barre verticale représentant la valeur du client pour cette variable.
    '''
    
    # Pour le moment j'effectue ici le sample.
    # dataRef = dataRef.sample(frac=0.01)
    
    x0 = dataRef[dataRef[target]==0][feature]
    x1 = dataRef[dataRef[target]==1][feature]
    del dataRef
    hist_data = [x0, x1]
    group_labels = ['Refusé', 'Accepté']
    fig = ff.create_distplot(hist_data, group_labels)
    fig.add_vline(x=valCust, line_width=3, line_dash="dash", line_color="red")
    return fig

# @st.cache(suppress_st_warning=True)
# def plotScatter2D(dataRef, listValCust):
    
#     # Pour le moment j'effectue ici le sample.
#     # dataRef = dataRef.sample(frac=0.001)
    
#     fig = px.scatter(
#         dataRef,
#         x=listValCust[0][0],
#         y=listValCust[1][0],
#         color=target
#         )
#     # fig.add_trace(px.scatter(x=listValCust[0][1], y=listValCust[1][1]))
#     fig.add_vline(x=listValCust[0][1], line_width=1, line_dash="solid", line_color="red")
#     fig.add_hline(y=listValCust[1][1], line_width=1, line_dash="solid", line_color="red")
    
#     fig.update_layout(showlegend=False)
    
#     return fig

@st.cache(suppress_st_warning=True)
def plotScatter2D(dataRef, listValCust):
    
    # Pour le moment j'effectue ici le sample.
    # dataRef = dataRef.sample(frac=0.001)
    
    fig = px.scatter(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        color=target
        )
    # fig.add_trace(px.scatter(x=listValCust[0][1], y=listValCust[1][1]))
    fig.add_vline(x=listValCust[0][1], line_width=1, line_dash="solid", line_color="red")
    fig.add_hline(y=listValCust[1][1], line_width=1, line_dash="solid", line_color="red")
    
    fig.update_layout(showlegend=True)
    
    return fig

@st.cache(suppress_st_warning=True)
def plotScatter3D(dataRef, listValCust):
    # Pour le moment j'effectue ici le sample.
    # dataRef = dataRef.sample(frac=0.001)
    
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
### Chart - End ###

### Others Function - Start ###
@st.cache(suppress_st_warning=True)
def getDFLocalFeaturesImportance(model,X,loanNumber,nbFeatures=12,inv=False):
    # X = X.sample(frac=0.01)
    idx = getTheIDX(data=X,columnName=loanColumn,value=loanNumber)
    shap_values = shap.TreeExplainer(model).shap_values(X.iloc[[idx]])[0]
    
    if inv:
        shap_values *= -1
    
    # if not inv:
    #     shap_values = shap.TreeExplainer(model).shap_values(X.iloc[[idx]])[0]
    # else:
    #     shap_values = shap.TreeExplainer(model).shap_values(X.iloc[[idx]])[0]
    
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

@st.cache(suppress_st_warning=True)
def loadDataAndModel():
    return pickle.load(open(os.getcwd()+'/pickle/dataRef.pkl', 'rb')),\
        pickle.load(open(os.getcwd()+'/pickle/dataCustomer.pkl', 'rb')),\
            pickle.load(open(os.getcwd()+'/pickle/model.pkl', 'rb'))

# @st.cache(suppress_st_warning=True)
# def loadDataAndModel():
#     return pickle.load(open(os.getcwd()+'\\pickle\\dataRef.pkl', 'rb')),\
#         pickle.load(open(os.getcwd()+'\\pickle\\dataCustomer.pkl', 'rb')),\
#             pickle.load(open(os.getcwd()+'\\pickle\\model.pkl', 'rb'))
### Others Function - End ###
            
### Model Prediction - Start ###
def modelPredict(data, model, loanNumber, threshold):
    '''
        Retourne la prédiction du modèle: 0 ou 1 en fonction du seuil
        ainsi que la valeur exact de probabilité donné par le modèle.
        Le score est modifié pour donner un score proche de 1 si acceptation du prêt.
        Cette correction est destiné à être plus compréhensible pour les clients.
    '''
    idx = getTheIDX(data=data,columnName=loanColumn,value=loanNumber)
    resultModel = model.predict_proba(data[data.index == idx])[:,1]
    resultModel = 1-resultModel
    return np.where(resultModel<threshold,0,1)[0],resultModel
### Model Prediction - End ###





# def showCustomerResult(model, data, threshold, idxCustomer=0):
    
#     # Trouver la plus proche valeur en Y à partir d'une valeur en X.
    
#     from collections import Counter
    
#     resultPredictProba = model.predict_proba(data)[:,1]
#     scoreCustomer = resultPredictProba[idxCustomer]
#     resultPredictProba.sort()
#     newIdxCustomer = np.where(resultPredictProba==scoreCustomer)[0][0]
    
#     newPredictProba = [round(i,2) for i in resultPredictProba]
#     newIdxC = np.where(newPredictProba==round(scoreCustomer,2))[0][0]
    
#     print(newIdxC)
    
#     dictTest = dict(Counter(newPredictProba))
#     xValues = list(dictTest.keys())
#     yValues = list(dictTest.values())
    
#     NEWIDX = np.where(xValues==round(newPredictProba[newIdxCustomer],2))[0][0]
#     print(xValues[NEWIDX])
#     print(yValues[NEWIDX])
    
#     idxMax = next(x[0] for x in enumerate(xValues) if x[1] > 0.51)

#     plt.figure(figsize=(20,10))
#     sns.lineplot(x=xValues[:idxMax], y=yValues[:idxMax], color='blue')
#     sns.lineplot(x=xValues[idxMax:], y=yValues[idxMax:], color='red')
#     plt.scatter(x=xValues[NEWIDX], y=yValues[NEWIDX], color='black')
    
#     plt.show()
#     st.pyplot()


            
def backgroundColor(codeHTML,color='000000',balise='span'):
    '''
    return a HTML code surounded by a span or div tag.
    The tag include a color for the background.
    '''
    return "<"+balise+" style='background-color:#"+color+";'>"+codeHTML+"</"+balise+">"