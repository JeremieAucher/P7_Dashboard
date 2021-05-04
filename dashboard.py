# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:39:47 2021

@author: Jérémie Aucher
"""

import streamlit as st
import utils
# import plotly.graph_objects as go
# import plotly.figure_factory as ff


def main():
    
    ########### Initialisation ##############################
    st.set_page_config(
        # layout="centered",
        layout='wide',
        initial_sidebar_state="collapsed"
        )
    threshold=0.51
    loanColumn = 'SK_ID_CURR'
    dataRef, dataCustomer, model = utils.loadDataAndModel()
    
    ########### Top ##############################
    col1, col2 = st.beta_columns((1,3))
    # col1.image('img/logo.png', width=150)
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
    
    # ########### Input ##############################
    # col1, col2 = st.beta_columns((1,2))
    # # col1, col2 = st.beta_columns((2))
    
    # with col1:
    #     st.text('')
    #     st.markdown('## Entrez le numéro de client:')
    # with col2:
    #     user_input = st.text_input('',value=list(dataCustomer[loanColumn])[1])
            

    ### Old Gestion d'Erreur
    nearest_value = min(list(dataCustomer[loanColumn]), key=lambda x:abs(x-int(user_input)))
    if not user_input == nearest_value:
        txt_error = 'Attention, numéro de client invalide.\n\nNuméro de client le plus proche: '+nearest_value
        st.warning(txt_error)
        user_input = nearest_value
        
    ########### Model Prediction ##############################
    predExact, predProba = utils.modelPredict(data=dataCustomer,
                                              model=model,
                                              loanNumber=int(user_input),
                                              threshold=threshold)

    ########### Model Prediction API ##########################    
    predExact, predProba = utils.modelPredict(data=dataCustomer,
                                              model=model,
                                              loanNumber=int(user_input),
                                              threshold=threshold)
    
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
        # st.markdown("## Score")
        

        predExact, predProba = utils.modelPredict(data=dataCustomer,
                                                  model=model,
                                                  loanNumber=int(user_input),
                                                  threshold=threshold)
        
        fig=utils.gauge_chart(predProba[0],threshold)
        st.write(fig)
    with col15:
        # Colonne vide pour centrer les elements
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
    
    # fig=utils.plotGlobalFeaturesImportance(model, dataRef, 10)
    # st.write(fig)
    
    # fig=utils.plotLocalFeaturesImportance(
    #     model=model,
    #     X=dataCustomer,
    #     loanNumber=int(user_input)
    #     )
    # st.write(fig)
    
    #### Analyse Mono & Bi variées
    ### Dist Plot
    col1, col2 = st.beta_columns((2))
    
    with col1:
        feature1 = st.selectbox('Choisissez la 1ère caractéristique:',df.index, index=0)
        valueCustomer1 = dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature1].values[0]
        fig = utils.plotDistOneFeature(dataRef, feature1, valueCustomer1)
        st.write(fig)
        # st.write(ff.create_distplot(
        #     [
        #         dataRef[dataRef['TARGET']==0][feature1][:615],
        #         dataRef[dataRef['TARGET']==1][feature1][:615]
        #     ],
        #     [
        #         'Refusé',
        #         'Accepté'
        #     ]))
        
    with col2:
        feature2 = st.selectbox('Choisissez la 2nd caractéristique:',df.index, index=1)
        valueCustomer2 = dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature2].values[0]
        fig = utils.plotDistOneFeature(dataRef, feature2, valueCustomer2)
        st.write(fig)
    
    #### Scatter Plot
    col1, col2 = st.beta_columns(2)
    ### Scatter Plot 2D
    with col1:
        # dictValueCustomer = {
        #     feature1:valueCustomer1,
        #     feature2:valueCustomer2
        # }
        listValueCustomer = [[feature1,valueCustomer1],[feature2,valueCustomer2]]
        fig = utils.plotScatter2D(dataRef, listValueCustomer)
        # st.markdown('## ')
        # st.markdown('## ')
        st.markdown('### ↓ Positionnement du client en fonction des 2 premières caractéristiques selectionnées')
        st.markdown('### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Positionnement du client en fonction des 3 caractéristiques selectionnées 🡮')
        st.write(fig)
    ### Scatter Plot 3D
    with col2:
        feature3 = st.selectbox('Choisissez la 3ème caractéristique:',df.index, index=2)
        listValueCustomer.append([feature3,dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature3].values[0]])
        fig = utils.plotScatter3D(dataRef, listValueCustomer)
        st.write(fig)
    
    
    
    ########### Sidebar ##############################
    # st.sidebar.markdown("# Validation de prêt")
    # st.sidebar.markdown("Entrez les informations clients\n")
    # user_input = st.sidebar.select_slider('Quelle est le numéro de prêt?',
    #                                       options=list(dataCustomer[loanColumn]))
    st.sidebar.title('Paramètres avancées')
    st.sidebar.header('Valeurs des 12 variables les plus importantes du client: ')
    
    df = utils.getDFLocalFeaturesImportance(model=model,
                                            X=dataCustomer,
                                            loanNumber=int(user_input),
                                            nbFeatures=12)
    for count, varName in enumerate(df.index):
        varValue = dataCustomer.loc[dataCustomer[loanColumn]==user_input, varName].values[0]
        st.sidebar.markdown(f'Variable n°{count+1}:')
        st.sidebar.markdown(f'- Nom: {varName}')
        st.sidebar.markdown(f'- Valeur: {varValue}')
        # st.sidebar.markdown(f'Variable n°{count+1}: {value}')
        # st.sidebar.markdown(f'Value n°{count+1}: {dataCustomer[dataCustomer[loanColumn]==user_input][value].values[0]}')
        # st.sidebar.markdown(f'Value n°{count+1}: {dataCustomer.loc[dataCustomer[loanColumn]==user_input, value].values[0]}')
    # Faire liste des variables globales les plus importantes :
    # Puis créer des selecteur pour les X variables les plus importantes
    # Indiquer à chaque fois pour chaque variable, la valeur du client conserné
    # CODE_GENDER
    


    ### Dist on 1st & 2nd most importante feature
    



# A Faire:
    # Ne conserver ici que les donnée test
    # Model et Donnée TRAIN à mettre uniquement côté API



if __name__ == "__main__":
    main()



# st.set_option('deprecation.showPyplotGlobalUse', False)




# loanColumn = 'SK_ID_CURR'
    
# dataRef, dataCustomer, model = utils.loadDataAndModel()


# loanNumberMin=dataCustomer['SK_ID_CURR'].min()
# loanNumberMax=dataCustomer['SK_ID_CURR'].max()




# predExact, predProba = utils.modelPredict(data=dataCustomer,
#                                           model=model,
#                                           loanNumber=int(user_input),
#                                           threshold=0.51)

# if predExact == 0:
#     predExact = 'Authorisé'
# else:
#     predExact = "Refusé"

# idxCustomer = utils.getTheIDX(dataCustomer,user_input,columnName='SK_ID_CURR')
# genderCustomer = utils.getGender(dataCustomer, idxCustomer)


# '# Dashboard'
# '## Sous titre'

# 'Numéro de Prêt: ', user_input
# 'Numéro de Client: ', idxCustomer
# 'Authorisation du prêt: ', predExact
# 'Résultat proba: ', round(float(predProba),3)
# 'Genre du client: ',genderCustomer
# utils.showCustomerResult(model, dataRef, 0.51, idxCustomer)
