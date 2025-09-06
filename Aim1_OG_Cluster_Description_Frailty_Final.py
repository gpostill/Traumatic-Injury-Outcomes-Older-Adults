#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 07:00:28 2024

@author: gepostill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from tableone import TableOne

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 



#Removing warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


##########################
#PLOTTING THE CLUSTERS 
##########################


def plot_tsne(df_train, cluster_labels_train, fig_label): 
    # Reducing the data wit t-SNE
    tsne = TSNE(n_components=2, perplexity=40, random_state=0)
    tsne_result = tsne.fit_transform(df_train)
    
    # Creating a scatter plot of t-SNE results colored by cluster
    plt.figure(figsize=(8, 6))
    #plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels_train, cmap='Paired')
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels_train, cmap='Paired')
    
    # Adding a legned with custom labels
    legend_labels = ['Cluster A', 'Cluster B', 'Cluster C']
    legend = plt.legend(*scatter.legend_elements())
    for text, label in zip(legend.get_texts(), legend_labels):
        text.set_text(label)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Cluster')
    plt.savefig(f'filepath/tsne_{fig_label}.png', dpi=700)
    plt.show()


def umap_plot(df_train_age, cluster_labels_train, label):

    # Compute UMPA Embeddings
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    # Need to check these hyperparameters
    embedding = umap_emb.fit_transform(df_train_age)
    
    # plot UMPA embeddings iwth clusters colored by subtype
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=cluster_labels_train, cmap='Paired', s=10)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Cluster')
    # plt.colorbar(label='Cluster')
    plt.savefig(f'filepath/umap_{label}.png', dpi=700)
    plt.show()



def pca_plot(df_train, clust_labels, label): 
    #Perform PCA 
    pca = PCA(n_components=2)   #Specifying two dimensinos
    pca_features = pca.fit_transform(df_train)          #check that this might need to be an array 
    
    #Create a Dataframe for the PCA components 
    pca_df = pd.DataFrame(data=pca_features, columns=['PCA1','PCA2'])
    pca_df['Cluster'] = clust_labels
    
    #Plot the clusters 
    plt.figure(figsize=(10,8))
    for cluster_id in pca_df['Cluster'].unique(): 
        cluster_data = pca_df[pca_df['Cluster'] == cluster_id]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster_id}', alpha=0.7)

    plt.title('Cluster Visualized with PCA', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.legend(title='Cluster')
    plt.savefig(f'filepath/PCA_{label}.png', dpi=700)
    plt.show()


def pca_plot_3D(df_train, clust_labels, label): 
    #Perform PCA 
    pca = PCA(n_components=3)   #Specifying two dimensinos
    pca_features = pca.fit_transform(df_train)          #check that this might need to be an array 
    
    #Create a Dataframe for the PCA components 
    pca_df = pd.DataFrame(data=pca_features, columns=['PCA1','PCA2','PCA3'])
    pca_df['Cluster'] = clust_labels
    
    #Plot the clusters 
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    #PLot each cluster in 3D space
    for cluster_id in pca_df['Cluster'].unique(): 
        cluster_data = pca_df[pca_df['Cluster'] == cluster_id]
        ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], cluster_data['PCA3'], label=f'Cluster {cluster_id}', alpha=0.7, s=60)

    ax.set_title('Cluster Visualized with PCA', fontsize=16)
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    ax.legend(title='Cluster')
    plt.savefig(f'filepath/PCA_3D_{label}.png', dpi=700)
    plt.show()


def create_heatmap(df, variables_list, figure_name): 
    #Copmpute the prevalence 
    prevalence = df.groupby('Cluster_Labels')[variables_list].mean()
    
    #Create the heatmap 
    plt.figure(figsize=(8,6))
    sns.heatmap(prevalence, cmap='Blues', annot=True, fmt='.2f', linewidth=0.5)
    
    #Customize plot 
    plt.xlabel("Condition")
    plt.ylabel("Clsuter")
    
    #Show and export the plot
    plt.tight_layout()
    plt.savefig(f'filepath/Aim1_{figure_name}.png', dpi=700)
    plt.show()    



def fix_dates(dataframe1):
         
    #Ensure date variables are in datetime format
    date_lists = ['indexdt', 'losseligdate_lookforward', 'last_elig_end', 'LTC_date', 'ltc_dt1', 'dthdate']
    for date in date_lists: 
        dataframe1[date] = pd.to_datetime(dataframe1[date])
    
    print(dataframe1['losseligdate_lookforward'].notna().sum())
    print(dataframe1['ltc_dt1'].notna().sum())
    print(dataframe1['dthdate'].notna().sum())
    
    #Create an event variable based on ltc_and death dates -  if LTC is earlier, take that date 
    dataframe1['event_occurred_dt'] = dataframe1['dthdate'].combine_first(dataframe1['ltc_dt1'])
    dataframe1['event_occurred_dt'] = dataframe1[['dthdate', 'ltc_dt1']].min(axis=1, skipna=True) 
    
    #Create an event occurrence indicator
    dataframe1['event_occurred'] = dataframe1['event_occurred_dt'].notnull().astype(int)
    
    #Create a censorhip date variables
    dataframe1['losseligdate_lookforward'] = dataframe1['losseligdate_lookforward'].fillna(pd.Timestamp('2024-03-31'))
    
    #Create an event of end variable
    dataframe1['event_or_end_dt'] = dataframe1[['event_occurred_dt', 'losseligdate_lookforward']].min(axis=1) 
   
    #Calculate time to event or end
    dataframe1['time_to_event'] = (dataframe1['event_or_end_dt'] - dataframe1['indexdt']).dt.days
    
    #Calcualte time to death 
    dataframe1['yrs_diff_death'] = (dataframe1['dthdate'] - dataframe1['indexdt']).dt.days / 365.25
    
    return dataframe1


def outcome_var(df):

    #   Alive and at home - 1yr, 2yr, 3yr, 4yr, 5yr
    
    #Time to ineligibillity 
    df['time_elig'] = (df['losseligdate_lookforward'] - df['indexdt']).dt.days / 365.25

    #Calculate time to event (ONLY)
    df['yrs_A_H'] = (df['event_occurred_dt'] - df['indexdt']).dt.days / 365.25
    
    #Binary columns for at leaast 1 to 5 years 
    var_names=["A_H_yr1", "A_H_yr2", "A_H_yr3", "A_H_yr4", "A_H_yr5"]
    alive_var_names=["alive_yr1", "alive_yr2", "alive_yr3", "alive_yr4", "alive_yr5"]
    years = [1, 2, 3, 4, 5]
    for alive_var, var, yr in zip(alive_var_names, var_names, years): 
        df[var] = np.where(df['yrs_A_H'] >= yr, 'Y', 'N')
        df[var] = np.where(df['time_elig'] < yr, np.nan, df[var]) #fill with ineligible if not eligible 
        #df[var] = np.where(df['time_elig'] < yr, 'Ineligible', df[var])#fill with ineligible if not eligible 
        print(df[var].value_counts(dropna=False))

        df[alive_var] = np.where(df['yrs_diff_death'] >= yr, 'Y', 'N')  #This var created elsewhere 
        df[alive_var] = np.where(df['time_elig'] < yr, np.nan, df[alive_var])#fill with ineligible if not eligible 
        #df[alive_var] = np.where(df['time_elig'] < yr, 'Ineligible', df[alive_var])#fill with ineligible if not eligible 
        print(df[alive_var].value_counts(dropna=False))

    return df



#########   IMPORTUNG THE MAIN DATASET (With and without imputation and scaing)  #########


## KPROTOYPE FRAILTY TRAINING DATASET (K=4)
df_kproto_frail_train = pd.read_csv('filename')
df_data_frail_train = pd.read_csv('filename')
frail_train_labels = df_data_frail_train['Cluster_Labels']

## KPROTOYPE FRAILTY TESTING DATASET (K=4)
#df_kproto_frail_test = pd.read_csv('filename')
#df_data_frail_test = pd.read_csv('filename')
#frail_test_labels = df_data_frail_test['Cluster_Labels']

## KPROTOYPE FRAILTY OVERALL DATASET (K=4)
df_kproto_frail = pd.read_csv('filename')
df_data_frail = pd.read_csv('filename')
frail_overall_labels = df_data_frail['Cluster_Labels']

#######

## KPROTOYPE NOT FRAIL TRAINING DATASET (K=4)
df_kproto_NOT_frail_train = pd.read_csv('filename')
df_data_NOT_frail_train = pd.read_csv('filename')
NOT_frail_train_labels = df_data_NOT_frail_train['Cluster_Labels']

## KPROTOYPE NOT FRAIL TESTING DATASET (K=4)
#df_kproto_NOT_frail_test = pd.read_csv('filename')
#df_data_NOT_frail_test = pd.read_csv('filename')
#NOT_frail_test_labels = df_data_NOT_frail_test['Cluster_Labels']

## KPROTOYPE NOT FRAIL OVERALL DATASET (K=4)
df_kproto_NOT_frail = pd.read_csv('filename')
df_data_NOT_frail = pd.read_csv('filename')
NOT_frail_overall_labels = df_data_NOT_frail['Cluster_Labels']

#######
exit()

#########   PLOTTING WITH TSNE  #########

#drop the labels on the dataframe - KPROTOTYPE 
df_kproto_frail_train = df_kproto_frail_train.drop('Cluster_Labels', axis=1)
#df_kproto_frail_test = df_kproto_frail_test.drop('Cluster_Labels', axis=1)
df_kproto_frail = df_kproto_frail.drop('Cluster_Labels', axis=1)

df_kproto_NOT_frail_train = df_kproto_NOT_frail_train.drop('Cluster_Labels', axis=1)
#df_kproto_NOT_frail_test = df_kproto_NOT_frail_test.drop('Cluster_Labels', axis=1)
df_kproto_NOT_frail = df_kproto_NOT_frail.drop('Cluster_Labels', axis=1)

#Plotting the tsne plots - KPROTOTYPE
plot_tsne(df_kproto_frail_train, frail_train_labels, 'OG_Frail_train')
#plot_tsne(df_kproto_frail_test, frail_test_labels, 'OG_Frail_test')
plot_tsne(df_kproto_frail, frail_overall_labels, 'OG_Frail_overall')

plot_tsne(df_kproto_NOT_frail_train, NOT_frail_train_labels, 'OG_NOT_Frail_train')
#plot_tsne(df_kproto_NOT_frail_test, NOT_frail_test_labels, 'OG_NOT_Frail_test')
plot_tsne(df_kproto_NOT_frail, NOT_frail_overall_labels, 'OG_NOT_Frail_overall')

#########   PLOTTING WITH UMAP  #########

#Run the plot for each training dataset - KPROTOTYPE  
umap_plot(df_kproto_frail_train, frail_train_labels, 'OG_Frail_train')
#umap_plot(df_kproto_frail_test, frail_test_labels, 'OG_Frail_test')
umap_plot(df_kproto_frail, frail_overall_labels, 'OG_Frail_overall')

umap_plot(df_kproto_NOT_frail_train, NOT_frail_train_labels, 'OG_NOT_Frail_train')
#umap_plot(df_kproto_NOT_frail_test, NOT_frail_test_labels, 'OG_NOT_Frail_test')
umap_plot(df_kproto_NOT_frail, NOT_frail_overall_labels, 'OG_NOT_Frail_overall')


#########   PLOTTING WITH PCA  #########

#Run the plot for each training dataset - KPROTOTYPE  
pca_plot(df_kproto_frail_train, frail_train_labels, 'OG_Frail_train')
#pca_plot(df_kproto_frail_test, frail_test_labels, 'OG_Frail_test')
pca_plot(df_kproto_frail, frail_overall_labels, 'OG_Frail_overall')

pca_plot(df_kproto_NOT_frail_train, NOT_frail_train_labels, 'OG_NOT_Frail_train')
#pca_plot(df_kproto_NOT_frail_test, NOT_frail_test_labels, 'OG_NOT_Frail_test')
pca_plot(df_kproto_NOT_frail, NOT_frail_overall_labels, 'OG_NOT_Frail_overall')


#########   PLOTTING WITH PCA in 3D     #########

#Run the plot for each training dataset - KPROTOTYPE  
pca_plot_3D(df_kproto_frail_train, frail_train_labels, 'OG_Frail_train')
#pca_plot_3D(df_kproto_frail_test, frail_test_labels, 'OG_Frail_test')
pca_plot_3D(df_kproto_frail, frail_overall_labels, 'OG_Frail_overall')

pca_plot_3D(df_kproto_NOT_frail_train, NOT_frail_train_labels, 'OG_NOT_Frail_train')
#pca_plot_3D(df_kproto_NOT_frail_test, NOT_frail_test_labels, 'OG_NOT_Frail_test')
pca_plot_3D(df_kproto_NOT_frail, NOT_frail_overall_labels, 'OG_NOT_Frail_overall')




#########   DESCRIPTIVE STATISTICS   #########

#The missing values in table one package for the following variables are giving me difficulty --> creating categorical variables
#'visit1_recent_los','visit2_los','visit3_los', 'visit1_days_since_dis']
def cat_admissions(value):
    if pd.isnull(value): 
        return 'No visit'
    elif value > 14: 
        return  '14+ days'
    else: 
        return "1-14 days"

for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:     #df_data_frail_test, df_data_NOT_frail_test
    #Apply to main dataframe
    #df['visit1_recent_los_cat'] = df['visit1_recent_los'].apply(cat_admissions)
    df['visit1_days_since_dis_cat'] = df['visit1_days_since_dis'].apply(cat_admissions)

def cat_rural(value):
    if pd.isnull(value): 
        return 'Missing'
    elif value >= 40: 
        return  'Rural'
    elif value < 40: 
        return  'Non-Rural'

for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:     #df_data_frail_test, df_data_NOT_frail_test
    #Apply to main dataframe
    #df['visit1_recent_los_cat'] = df['visit1_recent_los'].apply(cat_admissions)
    df['rural'] = df['rio2008'].apply(cat_rural)



#Categorize the number of visits
def cat_visits(value):
    if value == 0: 
        return 'No visit'
    elif value < 3: 
        return  '1-2 visits'
    elif value >= 3: 
        return  '3+ visits'
    else: 
        return "NA"
for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:     #df_data_frail_test, df_data_NOT_frail_test
    #apply to the main dataframe
    df['PCP_Visit_within_6mo_number_cat'] = df['PCP_Visit_within_6mo_number'].apply(cat_visits)
    df['Number_ED_NOT_Injury_6mo_cat'] = df['Number_ED_NOT_Injury_6mo'].apply(cat_visits)
    df['Number_ED_Injury_6mo_cat'] = df['Number_ED_Injury_6mo'].apply(cat_visits)
    df['num_mh_inpt_6mos_cat'] = df['num_mh_inpt_6mos'].apply(cat_visits)



#Reformat asthma
#for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:         #df_data_frail_test, df_data_NOT_frail_test
    #Convert to datetime
    #dates = ['asthma_date', 'copd_date']
    
    #Reformat the index date variable
    #for date in dates:
    #    df[date] = pd.to_datetime(df[date])
    
    #Reformat the index date variable
    #df['indexdt'] = pd.to_datetime(df['indexdt'])
    
    #days_with = ['asthma_days_with','copd_days_with']
    
    #for cond, days in zip(dates, days_with): 
    #    df[days] = (df['indexdt'] - df[cond]).dt.days


#Categorize the length of time with dx
def cat_conditions(value):
    if value == 0: 
        return  'Not Diagnosed'
    elif value <= 365: 
        return  'Year of Injury'
    elif value > 365: 
        return '>1 year ago'
    else: 
        return 'Not Diagnosed'

condiitons = ['asthma_days_with', 'copd_days_with', 'chf_days_with','cat_immune_def_days_with', 'odd_days_with', 'dementia_days_with', 'hyper_days_with', 'cat_tia_stroke_days_with', 
              'cat_cardiac_ischemic_days_with','cardiac_afib_days_with','cirrhosis_days_with', 'ckd_days_with','active_cancer_days_with', 'ocr_days_with',
              'cat_psoriasis_and_psoriatic_arthritis_dt', 'multiple_sclerosis_days_with','substance_abuse_days_with', 'cat_NM_disorder_days_with',
              'orad_days_with', 'osteoarthritis_days_with', 'osteoporosis_days_with', 'mood_dis_days_with', 'psych_dis_days_with']

for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:         #df_data_frail_test, df_data_NOT_frail_test
    #apply to the main dataframe
    for cond in condiitons:
        var = cond + "_cat"
        df[var] = df[cond].apply(cat_conditions)




variables = ['age', 'age_group', 'sex',"Recent_Immigrant","incquint", "age_labourforce_q_da", "material_resources_q_da", "racialized_NC_pop_q_da", "households_dwellings_q_da","rural","rio2008", 
             'hc_longterm', 'recent_chronic_dialysis', 'active_cancer', 'Prior_hipfx','HFRM_group',

             #'visit1_days_since_dis', 'Number_ED_NOT_Injury_6mo', 'Number_ED_Injury_6mo', 'PCP_Visit_within_6mo_number',  'num_mh_inpt_6mos', 
             'DAD_more_than_1', 'visit1_days_since_dis_cat', 'PCP_Visit_within_6mo_number_cat', 'Number_ED_NOT_Injury_6mo_cat', 'Number_ED_Injury_6mo_cat', 
             'num_mh_inpt_6mos_cat',
             "acg_score", 
            # 'asthma_days_with', 'copd_days_with', 'chf_days_with','cat_immune_def_days_with', 'odd_days_with', 'dementia_days_with', 'hyper_days_with', 'cat_tia_stroke_days_with', 
             #'cat_cardiac_ischemic_days_with','cardiac_afib_days_with','cirrhosis_days_with', 'ckd_days_with','active_cancer_days_with', 'ocr_days_with',
             #'cat_psoriasis_and_psoriatic_arthritis_dt', 'multiple_sclerosis_days_with','substance_abuse_days_with', 'cat_NM_disorder_days_with',
             #'orad_days_with', 'osteoarthritis_days_with', 'osteoporosis_days_with', 'mood_dis_days_with', 'psych_dis_days_with', 
             
             'asthma_days_with_cat', 'copd_days_with_cat', 'chf_days_with_cat','cat_immune_def_days_with_cat', 'odd_days_with_cat', 'dementia_days_with_cat', 'hyper_days_with_cat', 'cat_tia_stroke_days_with_cat', 
             'cat_cardiac_ischemic_days_with_cat','cardiac_afib_days_with_cat','cirrhosis_days_with_cat', 'ckd_days_with_cat','active_cancer_days_with_cat', 'ocr_days_with_cat',
             'cat_psoriasis_and_psoriatic_arthritis_dt_cat', 'multiple_sclerosis_days_with_cat','substance_abuse_days_with_cat', 'cat_NM_disorder_days_with_cat',
             'orad_days_with_cat', 'osteoarthritis_days_with_cat', 'osteoporosis_days_with_cat', 'mood_dis_days_with_cat', 'psych_dis_days_with_cat', 
             
             'N_active_drug_cat','ANTIDEPRESSANT_2yr', 'ANTIPSYCHOTIC_2yr', 'CNS_STIMULANT_2yr', 'ANXIOLYTICS_SEDATIVES_2yr',
             'OPIOIDS_2yr', 'ANTI_HYPERGLYCEMICS_2yr', 'INSULIN_2yr', 'ANTI_HYPERTENSIVES_2yr', 'ANTI_COAGULANTS_2yr', 'DEMENTIA_2yr', 'ANTICHOLINERGIC_2yr', 'ANTIOSTEOPOROSIS_2yr',

             'ISS_calcD','Severe_Head','Severe_Chest','Severe_Abdomen', 'Severe_Neck','Severe_Spine','Severe_Extern','Severe_Face','Severe_UprExtr','Severe_LwrExtr', 'Un_Intetnional',
                       
             ]


#Manually specifying the data types 
nonnormal = ['age','ISS_calcD',"rio2008", "acg_score", 'N_active_drug_cat']
             #'Number_ED_NOT_Injury_6mo', 'Number_ED_Injury_6mo', 'PCP_Visit_within_6mo_number',  'num_mh_inpt_6mos', 'visit1_days_since_dis',
             #'asthma_days_with', 'copd_days_with', 'chf_days_with','cat_immune_def_days_with', 'odd_days_with', 'dementia_days_with', 'hyper_days_with', 'cat_tia_stroke_days_with', 'cat_cardiac_ischemic_days_with',
             #'cardiac_afib_days_with','cirrhosis_days_with', 'ckd_days_with','active_cancer_days_with', 'ocr_days_with',
             #'cat_psoriasis_and_psoriatic_arthritis_dt', 'multiple_sclerosis_days_with','substance_abuse_days_with', 'cat_NM_disorder_days_with',
             #'orad_days_with', 'osteoarthritis_days_with', 'osteoporosis_days_with', 'mood_dis_days_with', 'psych_dis_days_with']



categorical = ['age_group','sex',"rural","incquint", "age_labourforce_q_da", "material_resources_q_da", "racialized_NC_pop_q_da", "households_dwellings_q_da",
               'hc_longterm', 'recent_chronic_dialysis', 'active_cancer', 'DAD_more_than_1','Prior_hipfx','HFRM_group',"Recent_Immigrant",
               'visit1_days_since_dis_cat','PCP_Visit_within_6mo_number_cat', 'Number_ED_NOT_Injury_6mo_cat', 'Number_ED_Injury_6mo_cat', 
               'num_mh_inpt_6mos_cat',
               
               'asthma_days_with_cat', 'copd_days_with_cat', 'chf_days_with_cat','cat_immune_def_days_with_cat', 'odd_days_with_cat', 'dementia_days_with_cat', 'hyper_days_with_cat', 'cat_tia_stroke_days_with_cat', 
               'cat_cardiac_ischemic_days_with_cat','cardiac_afib_days_with_cat','cirrhosis_days_with_cat', 'ckd_days_with_cat','active_cancer_days_with_cat', 'ocr_days_with_cat',
               'cat_psoriasis_and_psoriatic_arthritis_dt_cat', 'multiple_sclerosis_days_with_cat','substance_abuse_days_with_cat', 'cat_NM_disorder_days_with_cat',
               'orad_days_with_cat', 'osteoarthritis_days_with_cat', 'osteoporosis_days_with_cat', 'mood_dis_days_with_cat', 'psych_dis_days_with_cat', 

               'Severe_Head','Severe_Chest','Severe_Abdomen', 'Severe_Neck','Severe_Spine','Severe_Extern','Severe_Face','Severe_UprExtr','Severe_LwrExtr', 'Un_Intetnional',
               'ANTIDEPRESSANT_2yr', 'ANTIPSYCHOTIC_2yr', 'CNS_STIMULANT_2yr', 'ANXIOLYTICS_SEDATIVES_2yr',
               'OPIOIDS_2yr', 'ANTI_HYPERGLYCEMICS_2yr', 'INSULIN_2yr', 'ANTI_HYPERTENSIVES_2yr', 'ANTI_COAGULANTS_2yr', 'DEMENTIA_2yr', 'ANTICHOLINERGIC_2yr', 'ANTIOSTEOPOROSIS_2yr'
                     ]

#df_data_NOT_frail_train['incquint'] = df_data_NOT_frail_train['incquint'].astype(int)

for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:  #df_data_frail_test, df_data_NOT_frail_test
    #Recode the variables that will have mized types - MAIN DATAFRAME
    #df['incquint'] = df['incquint'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
    #df['incquint'] = df['incquint'].replace(("1.0","2.0","3.0","4.0","5.0"), ('Q1','Q2','Q3','Q4','Q5'))
    df['age_labourforce_q_da'] = df['age_labourforce_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
    df['material_resources_q_da'] = df['material_resources_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
    df['racialized_NC_pop_q_da'] = df['racialized_NC_pop_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
    df['households_dwellings_q_da'] = df['households_dwellings_q_da'].replace((1,2,3,4,5), ('Q1','Q2','Q3','Q4','Q5'))
    df['HFRM_group'] = df['HFRM_group'].replace((2,3,4,5,6,7), ('Group2','Group3','Group4','Group5','Group6','Group7'))
    df['Cluster_Labels'] = df['Cluster_Labels'].replace((0,1,2,3,4,5,6,7), ('Cluster 0','Cluster 1','Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'))

for df in [df_data_frail_train, df_data_frail, df_data_NOT_frail_train, df_data_NOT_frail]:     #df_data_frail_test, df_data_NOT_frail_test
    for cat in categorical: 
        df[cat] = df[cat].replace((0,1), ("N", "Y"))

# Create TableOne Object - stratifying by Cluster


#Frail Training data
table1_frail_train = TableOne(df_data_frail_train, columns=variables, categorical=categorical, nonnormal=nonnormal,  missing=True, groupby='Cluster_Labels', pval=True)
table1_frail_train.to_csv('filepath/filename.csv')

#Frail Testing data
#table1_frail_test = TableOne(df_data_frail_test, columns=variables, categorical=categorical, nonnormal=nonnormal,  missing=True, groupby='Cluster_Labels', pval=True)
#table1_frail_test.to_csv('/linux_home/gepostill/Files/u/gepostill/EXPORT - Brandon/FrailtyClusters/Table1_Kproto_OG_Frail_Test.csv')

#Frail Overall data
table1_frail_overall = TableOne(df_data_frail, columns=variables, categorical=categorical, nonnormal=nonnormal,  missing=True, groupby='Cluster_Labels', pval=True)
table1_frail_overall.to_csv('filepath/filename.csv')

#Not Frail Training data
table1_NOT_frail_train = TableOne(df_data_NOT_frail_train, columns=variables, categorical=categorical, nonnormal=nonnormal,  missing=True, groupby='Cluster_Labels', pval=True)
table1_NOT_frail_train.to_csv('filepath/filename.csv')

#Not Frail Testing data
#table1_NOT_frail_test = TableOne(df_data_NOT_frail_test, columns=variables, categorical=categorical, nonnormal=nonnormal,  missing=True, groupby='Cluster_Labels', pval=True)
#table1_NOT_frail_test.to_csv('filepath/filename.csv')

#Not Frail Overall data
table1_NOT_frail_overall = TableOne(df_data_NOT_frail, columns=variables, categorical=categorical, nonnormal=nonnormal,  missing=True, groupby='Cluster_Labels', pval=True)
table1_NOT_frail_overall.to_csv('filepath/filename.csv')



exit()


