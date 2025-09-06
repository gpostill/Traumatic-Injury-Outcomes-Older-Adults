#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:19:49 2025

@author: gepostill
"""


#################################
# IMPORTING PACKAGES
################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tableone import TableOne

from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
# from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import OPTICS

from sklearn.model_selection import train_test_split
from keras import layers, models 
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder


import warnings
warnings.filterwarnings('ignore')



#################################
# DEFINING RELEVANT FUNCTIONS 
################################


#Create a function to determine the optimal K 
def find_optimal_k(data, max_k, file_name, categorical_indices=None): 
    costs=[]
    silhouette_scores=[]
    davies_bouldin_scores=[]
    calinski_harabasz_scores=[]
    #Iterate through the possible values
    for k in range(9, max_k+1): 
        kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=-1, random_state=42)
        clusters = kproto.fit_predict(data, categorical=categorical_indices)
        costs.append(kproto.cost_)  #Cost funciton value 
        silhouette_scores.append(silhouette_score(data, clusters))
        calinski_harabasz_scores.append(calinski_harabasz_score(data, clusters))
        davies_bouldin_scores.append(davies_bouldin_score(data, clusters))
        
    ##exporting a dataframe with the cluster values
    data = {'K': list(range(9,max_k+1)), "Cost": costs, "Silhouette": silhouette_scores, "Davies-Bouldin": davies_bouldin_scores, "Calinski-Harabaz": calinski_harabasz_scores}
    df_temp_scores = pd.DataFrame(data)
    df_temp_scores.to_csv(f'/file_path/Aim1_KProto_scores_{file_name}.csv', index=False)

        
    return costs, silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores



#Create a function that runs the images 
def plot_K_scores(K_Max, costs, silhouettte_scores, calinski_harabasz_scores, davies_bouldin_scores, fig_name): 
    #Define the range of K values
    K_values = range(2, K_Max+1)
    plt.figure(figsize=(12,8))
    #Bic Subplot
    plt.subplot(2,2,1)
    plt.plot(K_values, costs, label='Costs', marker='.')
    plt.xlabel('Number of Classes (K)')
    plt.ylabel('Costs')
    plt.title('Costs')
    #Davies-Bouldin
    plt.subplot(2,2,2)
    plt.plot(K_values, davies_bouldin_scores, label='Davies-Bouldin', marker='.')
    plt.xlabel('Number of Classes (K)')
    plt.ylabel('Davies-Bouldin')
    plt.title('Davies-Bouldin')
    #Silhouette Score Subplot
    plt.subplot(2,2,3)
    plt.plot(K_values, silhouettte_scores, label='Silhouette Score', marker='.')
    plt.xlabel('Number of Classes (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    #Calinski-Harabasz
    plt.subplot(2,2,4)
    plt.plot(K_values, calinski_harabasz_scores, label='Calinski-Harabasz', marker='.')
    plt.xlabel('Number of Classes (K)')
    plt.ylabel('Calinski-Harabasz')
    plt.title('Calinski-Harabasz')
    #Final plot parameters
    plt.tight_layout()
    plt.savefig(f'/file_path/Aim1_KProto_{fig_name}.png', dpi=700)
    plt.show()    


#Define a function to build and train an autoencoder for a given encoding size 
def train_autoencoder(input_dim, encoding_dim, X_train, X_val, epochs=50, batch_size=32): 
    input_layer = layers.Input(shape=(input_dim, ))
    #Encoder 
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    #Decoder
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)    #Sigmoid was chosen vs. linear because output bounded by 0/1
    #Autoencoder model 
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    #Train encoder 
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, 
                              shuffle=True, 
                              validation_data=(X_val, X_val), 
                              verbose=0)
    #Get final validaiton loss
    val_loss = history.history['val_loss'][-1]
    return val_loss


#################################
# IMPORTING DATAFRAME
################################

df = pd.read_csv('/file_path/file_name.csv')

df_original = df.copy()
            
################################################
# FEATURE SELECTION FOR CLUCTERING
################################################


#Recode the area-level social categorical variables 
#df['incquint'] = df['incquint'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
#df['age_labourforce_q_da'] = df['age_labourforce_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
#df['material_resources_q_da'] = df['material_resources_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
#df['racialized_NC_pop_q_da'] = df['racialized_NC_pop_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
#df['households_dwellings_q_da'] = df['households_dwellings_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))

#  Recoding the values of missing in quintiles 
for var in ['incquint', 'age_labourforce_q_da', 'material_resources_q_da', 'racialized_NC_pop_q_da', 'households_dwellings_q_da']:
    df[var] = df[var].fillna('Missing')

#   Specify the features used for clustering
clustering_features = ['age','visit1_days_since_dis',"rio2008", "acg_score", 'N_active_drug_cat',
                       'Number_ED_NOT_Injury_6mo', 'Number_ED_Injury_6mo', 'PCP_Visit_within_6mo_number',  'num_mh_inpt_6mos', 
                       'asthma_days_with', 'copd_days_with', 'chf_days_with','cat_immune_def_days_with', 'odd_days_with', 'dementia_days_with', 'hyper_days_with', 'cat_tia_stroke_days_with', 'cat_cardiac_ischemic_days_with',
                       'cardiac_afib_days_with','cirrhosis_days_with', 'ckd_days_with','active_cancer_days_with', 'ocr_days_with',
                       'multiple_sclerosis_days_with','substance_abuse_days_with', 'cat_NM_disorder_days_with',
                       'orad_days_with', 'osteoarthritis_days_with', 'osteoporosis_days_with', 'mood_dis_days_with', 'psych_dis_days_with', 
                       
                       'sex',"incquint", "age_labourforce_q_da", "material_resources_q_da", "racialized_NC_pop_q_da", "households_dwellings_q_da",
                       'hc_longterm', 'recent_chronic_dialysis', 'active_cancer', 'DAD_more_than_1','Prior_hipfx','HFRM_group',"Recent_Immigrant",
                       'ANTIDEPRESSANT_2yr', 'ANTIPSYCHOTIC_2yr', 'CNS_STIMULANT_2yr', 'ANXIOLYTICS_SEDATIVES_2yr',
                       'OPIOIDS_2yr', 'ANTI_HYPERGLYCEMICS_2yr', 'INSULIN_2yr', 'ANTI_HYPERTENSIVES_2yr', 'ANTI_COAGULANTS_2yr', 'DEMENTIA_2yr', 'ANTICHOLINERGIC_2yr', 'ANTIOSTEOPOROSIS_2yr'
                    
                       ]

#Make a copy of the dataset 
df_original = df.copy()

#Subset to the variables used in the clustering
df = df[clustering_features]
#Note: I droppped:
        #   'frailty', 'visit1_recent_los', 'visit1_recent_dx10code1', 'cat_psoriasis_and_psoriatic_arthritis_dt',  'moi',
        #   'ISS_calcD', 'Severe_Head','Severe_Chest','Severe_Abdomen', 'Severe_Neck','Severe_Spine','Severe_Extern','Severe_Face','Severe_UprExtr','Severe_LwrExtr', 'Un_Intetnional',



#Manually specifying the data types 
numerical_cols = ['age','visit1_days_since_dis',"rio2008", "acg_score", 'N_active_drug_cat',
                  'Number_ED_NOT_Injury_6mo', 'Number_ED_Injury_6mo', 'PCP_Visit_within_6mo_number',  'num_mh_inpt_6mos', 
                  'asthma_days_with', 'copd_days_with', 'chf_days_with','cat_immune_def_days_with', 'odd_days_with', 'dementia_days_with', 'hyper_days_with', 'cat_tia_stroke_days_with', 'cat_cardiac_ischemic_days_with',
                  'cardiac_afib_days_with','cirrhosis_days_with', 'ckd_days_with','active_cancer_days_with', 'ocr_days_with',
                  'multiple_sclerosis_days_with','substance_abuse_days_with', 'cat_NM_disorder_days_with',
                  'orad_days_with', 'osteoarthritis_days_with', 'osteoporosis_days_with', 'mood_dis_days_with', 'psych_dis_days_with']



categorical_cols = ['sex',"incquint", "age_labourforce_q_da", "material_resources_q_da", "racialized_NC_pop_q_da", "households_dwellings_q_da",
                    'hc_longterm', 'recent_chronic_dialysis', 'active_cancer', 'DAD_more_than_1','Prior_hipfx','HFRM_group',"Recent_Immigrant",
                    'ANTIDEPRESSANT_2yr', 'ANTIPSYCHOTIC_2yr', 'CNS_STIMULANT_2yr', 'ANXIOLYTICS_SEDATIVES_2yr',
                    'OPIOIDS_2yr', 'ANTI_HYPERGLYCEMICS_2yr', 'INSULIN_2yr', 'ANTI_HYPERTENSIVES_2yr', 'ANTI_COAGULANTS_2yr', 'DEMENTIA_2yr', 'ANTICHOLINERGIC_2yr', 'ANTIOSTEOPOROSIS_2yr'
                     ]


#################################
# PRE-PROCESSING
################################

#   NORMALIZATION: Scaling / normalizing the numerical columns for better convergence
scaler2 = StandardScaler()
# Scales each column separately, normalizing around column mean
df[numerical_cols] = scaler2.fit_transform(df[numerical_cols])

#   Recode Sex
df['sex'] = df['sex'].replace(('M','F'), (1, 0))

#  Recoding the Y/N columns to 0/1
df[categorical_cols] = df[categorical_cols].replace(('Y','N'), (1, 0))

# Any blank not yet dealt with will be filled with NA (this primarily applies to the numerical column -- age)
df = df.fillna(-5)
df = df.replace('Missing', -5)


#################################
# K-PROTOTYPE CLUSTERING
################################


#Identify the indices of the categorical columns 
categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]

#Convert the training data to an array 
array_kproto = df.to_numpy()

#Find the optimal K
max_k = 10
costs_train, silhouette_scores_train, DB_scores_train, CH_scores_train = find_optimal_k(array_kproto, max_k=max_k, file_name="OG_NOT_Frail_training_Round2", categorical_indices=categorical_indices)
#OG_NOT_Frail_training
#OG_NOT_Frail_training_Round2

#Import the prior scores 
prior_scores = pd.read_csv('/file_path/file_name.csv')
prior_costs_train = prior_scores['Cost']
prior_silhouette_scores_train = prior_scores['Silhouette']
prior_DB_scores_train = prior_scores['Davies-Bouldin']
prior_CH_scores_train = prior_scores['Calinski-Harabaz']

#Revise the training scores 
costs_train = prior_costs_train.to_list() + costs_train
silhouette_scores_train = prior_silhouette_scores_train.to_list() + silhouette_scores_train
DB_scores_train = prior_DB_scores_train.to_list() + DB_scores_train
CH_scores_train = prior_CH_scores_train.to_list() + CH_scores_train

#Plot the results to determine the optimal K 
plot_K_scores(max_k, costs_train, silhouette_scores_train, CH_scores_train, DB_scores_train, 'OG_NOT_Frail_train')


#######################################
#   K = 4
#######################################

#Specify the optimal number
optimal_k=3

#Fit with the optimal number of clusters 
kproto = KPrototypes(n_clusters=optimal_k, init='Cao', n_init=20, n_jobs=-1, random_state=42)
print("Fitting final clusters:")
kproto.fit(array_kproto, categorical=categorical_indices)

#Label the original and the cluster DataFrame 
print("Extracting labels from clusters:")
original_labels = kproto.labels_
#Label the dimension reduced dataframe
df_array_kproto = pd.DataFrame(array_kproto)
df_array_kproto['Cluster_Labels'] = original_labels
df_array_kproto.to_csv('file_path/file_name.csv', index=False)
#Label the original dataframe
df_original['Cluster_Labels'] = original_labels
df_original.to_csv('/file_path/file_name.csv', index=False)

print("Evaluating final clusters:")

#Calculating the score values 
cost = kproto.cost_
print(f"The cost : {kproto.cost_}")
sil_score = silhouette_score(array_kproto, original_labels)
print(f"The silhouette_scores : {sil_score}")
ch_score = calinski_harabasz_score(array_kproto, original_labels)
print(f"The calinski_harabasz_scores : {ch_score}")
db_score = davies_bouldin_score(array_kproto, original_labels)
print(f"The davies_bouldin_scores: {db_score}")

#Performance of final clusters
data = {'K': [3], "Cost": [cost], "Silhouette": [sil_score], "Davies-Bouldin": [db_score], "Calinski-Harabaz": [ch_score]}
df_final_scores = pd.DataFrame(data)
df_final_scores.to_csv('/file_path/file_name.csv', index=False)



