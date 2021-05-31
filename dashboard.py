# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:39:47 2021

@author: Jérémie Aucher
"""

import streamlit as st
import utils

def main():
    
    ########### Initialisation ##############################
    st.set_page_config(
        # layout="centered",
        layout='wide',
        initial_sidebar_state="collapsed"
        )
    loanColumn = 'SK_ID_CURR'
    dataRef, dataCustomer = utils.loadData()
    model = utils.loadModel()
    threshold = utils.loadThreshold()
    
    ########### Top ##############################
    col1, col2 = st.beta_columns((1,3))
    with col1:
        st.image('img/logo.png', width=300)
    with col2:
        st.title('Simulation de prêt')
        st.header('Obtenez une réponse instantanément')
        ### Input ###
        user_input = st.selectbox('Entrez le numéro de prêt:',dataCustomer[loanColumn].tolist())
        idxCustomer = utils.getTheIDX(dataCustomer,user_input,loanColumn)
        'Vous avez selectionné le prêt n°: ', user_input,' correspondant au client n°',idxCustomer
        ### DF des Local Features Importance
        df = utils.getDFLocalFeaturesImportance(model=model,
                                        X=dataCustomer,
                                        loanNumber=int(user_input),
                                        nbFeatures=12)

    ########### Model Prediction API ##########################
    predExact, predProba = utils.apiModelPrediction(data=dataCustomer,
                                                    loanNumber=int(user_input))
    
    ########### Loan Validation ##############################
    st.markdown("# Validation du prêt")
    loanResult = 'Status du prêt: '
    if predExact:
        loanResult += "Validé !"
        st.success(loanResult)
    else:
        loanResult += "Refusé..."
        st.error(loanResult)
    
    
    ########### Core ##############################
    col1, col15, col2 = st.beta_columns((2,1,2))
    with col1:
        ### Gauge Score
        fig=utils.gauge_chart(predProba,threshold)
        st.write(fig)
    with col15:
        # Empty column to center the elements
        st.write("")
    with col2:
        ### Img OK/NOK
        if predExact:
            st.image('img/ok.png', width=400)
        else:
            st.image('img/nok.png', width=450)
    
    ### Global & Local Features Importance
    col1, col2 = st.beta_columns((2))
    ### Col 1/2 ### Global Features Importance
    with col1:
        fig=utils.plotGlobalFeaturesImportance(model, dataRef, 10)
        st.write(fig)            
    ### Col 2/2 ### Local Features Importance
    with col2:
        fig=utils.plotLocalFeaturesImportance(
            model=model,
            X=dataCustomer,
            loanNumber=int(user_input)
            )
        st.write(fig)
    
    #### Mono & Bi analysis
    ### Dist Plot
    col1, col2 = st.beta_columns((2))
    
    with col1:
        feature1 = st.selectbox('Choisissez la 1ère caractéristique:',df.index, index=0)
        valueCustomer1 = dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature1].values[0]
        fig = utils.plotDistOneFeature(dataRef, feature1, valueCustomer1)
        st.write(fig)
        
    with col2:
        feature2 = st.selectbox('Choisissez la 2nd caractéristique:',df.index, index=1)
        valueCustomer2 = dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature2].values[0]
        fig = utils.plotDistOneFeature(dataRef, feature2, valueCustomer2)
        st.write(fig)
    
    #### Scatter Plot
    col1, col2 = st.beta_columns(2)
    ### Scatter Plot 2D
    with col1:
        listValueCustomer = [[feature1,valueCustomer1],[feature2,valueCustomer2]]
        fig = utils.plotScatter2D(dataRef, listValueCustomer)
        st.markdown('### ↓ Positionnement du client en fonction des 2 premières caractéristiques selectionnées')
        st.markdown('### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Positionnement du client en fonction des 3 caractéristiques selectionnées 🡮')
        st.write(fig)
    ### Scatter Plot 3D
    with col2:
        feature3 = st.selectbox('Choisissez la 3ème caractéristique:',df.index, index=2)
        listValueCustomer.append([feature3,dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature3].values[0]])
        fig = utils.plotScatter3D(dataRef, listValueCustomer)
        st.write(fig)

if __name__ == "__main__":
    main()
