#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 08:25:51 2025

@author: gepostill
"""


######################################
#IMPORTING THE NECESSSARY PACKAGES 
######################################


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sas7bdat import SAS7BDAT
from tableone import TableOne
from itertools import cycle 
import os
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter

#Removing warnings for clearner output
import warnings
warnings.filterwarnings('ignore')


######################################
#DEFINING THE RELEVANT FUNCTIONS
######################################


#Categorize the number of visits
def cat_ISS(value):
    if value < 16: 
        return 'ISS <16'
    elif value >= 16: 
        return  'ISS >=16'
    else: 
        return "NA"


def clean_data(df):

    #Subsetting the data to the relevant column
    rel_col = ['indexdt', 'losseligdate_lookforward', 'hcd_admdate', 'cpro_eligdt', 'ccrs_admdate', 'dthdate', 'aminst_traumacnt',  
               'sex', 'frailty', 'age', 'age_group', 'ISS_calcD', 'Severe_Head', 'Cluster_Labels']
    df = df[rel_col]
         
    #Relabel trauma center 
    df['aminst_traumacnt'] = df['aminst_traumacnt'].replace((0,1), ('Non-TC','TC'))

    #Categorize ISS 
    df['ISS_cat'] = df['ISS_calcD'].apply(cat_ISS)
    
    #Ensure date variables are in datetime format
    date_lists = ['indexdt', 'losseligdate_lookforward', 'hcd_admdate', 'cpro_eligdt', 'ccrs_admdate', 'dthdate']
    for date in date_lists: 
        df[date] = pd.to_datetime(df[date])

    #Calculate teh date admitted to LTC 
    LTC_dates = ['hcd_admdate', 'cpro_eligdt', 'ccrs_admdate']
    df['LTC_date'] = df[LTC_dates].min(axis=1)

    
    print(df['losseligdate_lookforward'].notna().sum())
    print(df['LTC_date'].notna().sum())
    print(df['dthdate'].notna().sum())
    
    #Create an event variable based on ltc_and death dates -  if LTC is earlier, take that date 
    df['event_occurred_dt'] = df['dthdate'].combine_first(df['LTC_date'])
    df['event_occurred_dt'] = df[['dthdate', 'LTC_date']].min(axis=1, skipna=True) 
    
    #Create an event occurrence indicator
    df['event_occurred'] = df['event_occurred_dt'].notnull().astype(int)
    
    #Create a censorhip date variables
    df['losseligdate_lookforward'] = df['losseligdate_lookforward'].fillna(pd.Timestamp('2024-03-31'))
    
    #Create an event of end variable
    df['event_or_end_dt'] = df[['event_occurred_dt', 'losseligdate_lookforward']].min(axis=1) 
   
    #Calculate time to event or end
    df['time_to_event'] = (df['event_or_end_dt'] - df['indexdt']).dt.days
    df['time_to_event'] = df['time_to_event'] / 365.25
    #Calcualte time to death 
    df['yrs_diff_death'] = (df['dthdate'] - df['indexdt']).dt.days / 365.25

    ##### For tabulating the outcomes 
    #Calculate eligible time (in years)
    df['time_eligible_yr'] = (df['losseligdate_lookforward'] - df['indexdt']).dt.days / 365.25


    return df 




def survival_var(df):

    #Create an event occurrence indicator
    df['death_occurred'] = df['dthdate'].notnull().astype(int)
        
    #Create an event of end variable
    df['death_or_end_dt'] = df[['dthdate', 'losseligdate_lookforward']].min(axis=1) 
   
    #Calculate time to event or end/censorship
    df['time_to_death'] = (df['death_or_end_dt'] - df['indexdt']).dt.days
    df['time_to_death'] = df['time_to_death'] / 365.25

    return df 



def kaplan_analysis(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    #Plot each cluster
    for i, cluster in enumerate(unique_clusters): 
        cluster_data = df[df[cc_Var] == cluster]
        kmf.fit(durations=cluster_data['time_to_event'], event_observed=cluster_data['event_occurred'])
        kmf.plot_survival_function(ci_show=True, color=cmap(i), label = f'Cluster {cluster}')
    #plt.xlabel('Days Post Injury')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    #plt.xlim((0,2000))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Alive and at Home Probability')
    plt.savefig(f'file _path/Kaplan_{version}.png', dpi=700)
    plt.show()
    
    return 


def kaplan_analysis_death(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    #Plot each cluster
    for i, cluster in enumerate(unique_clusters): 
        cluster_data = df[df[cc_Var] == cluster]
        kmf.fit(durations=cluster_data['time_to_death'], event_observed=cluster_data['death_occurred'])
        kmf.plot_survival_function(ci_show=True, color=cmap(i), label = f'Cluster {cluster}')
    #plt.title('Kaplan-Meier Curves by Cluster')
    plt.xlabel('Years After Injury')
    #plt.xlim(3400)
    plt.ylim((0,1.0))
    #plt.xlim((0,2000))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Survival Probability')
    plt.savefig(f'/filepath/Kaplan_death_{version}.png', dpi=700)
    plt.show()
    
    return 

def kaplan_analysis_TC(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['aminst_traumacnt'].unique()[0]: '-', #Solid line 
                 df['aminst_traumacnt'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['aminst_traumacnt'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['aminst_traumacnt'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_event'], event_observed=cluster_tc_data['event_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    #plt.xlim((0,2000))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Alive and at Home Probability')
    plt.savefig(f'filepath/Kaplan_TC_{version}.png', dpi=700)
    plt.show()
    
    return 


def kaplan_analysis_TC_death(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['aminst_traumacnt'].unique()[0]: '-', #Solid line 
                 df['aminst_traumacnt'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['aminst_traumacnt'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['aminst_traumacnt'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_death'], event_observed=cluster_tc_data['death_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Survival Probability')
    plt.savefig(f'filepath/Kaplan_TC_death_{version}.png', dpi=700)
    plt.show()
    
    return 

def kaplan_analysis_sex(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['sex'].unique()[0]: '-', #Solid line 
                 df['sex'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['sex'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['sex'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_event'], event_observed=cluster_tc_data['event_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Alive and at Home Probability')
    plt.savefig(f'/filepath/Kaplan_sex_{version}.png', dpi=700)
    plt.show()
    
    return 


def kaplan_analysis_sex_death(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['sex'].unique()[0]: '-', #Solid line 
                 df['sex'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['sex'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['sex'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_death'], event_observed=cluster_tc_data['death_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Survival Probability')
    plt.savefig(f'/filepath/Kaplan_sex_death_{version}.png', dpi=700)
    plt.show()
    
    return 


def kaplan_analysis_head(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['Severe_Head'].unique()[0]: '-', #Solid line 
                 df['Severe_Head'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['Severe_Head'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['Severe_Head'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_event'], event_observed=cluster_tc_data['event_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Alive and at Home Probability')
    plt.savefig(f'/filepath/Kaplan_head_{version}.png', dpi=700)
    plt.show()
    
    return 


def kaplan_analysis_head_death(df, cc_Var, version, fontsize=6, ncol=1): 
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['Severe_Head'].unique()[0]: '-', #Solid line 
                 df['Severe_Head'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['Severe_Head'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['Severe_Head'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_death'], event_observed=cluster_tc_data['death_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Alive and at Home Probability')
    plt.savefig(f'/filepath/Kaplan_head_death_{version}.png', dpi=700)
    plt.show()
    
    return 



def kaplan_analysis_ISS(df, cc_Var, version, fontsize=6, ncol=1): 
    
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['ISS_cat'].unique()[0]: '-', #Solid line 
                 df['ISS_cat'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['ISS_cat'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['ISS_cat'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_event'], event_observed=cluster_tc_data['event_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol)
    plt.ylabel('Alive and at Home Probability')
    plt.savefig(f'/filepath/Kaplan_ISS_{version}.png', dpi=700)
    plt.show()
    
    return 


def kaplan_analysis_ISS_death(df, cc_Var, version, fontsize=6, ncol=1):  #loc='best', bbox_to_anchor=None
    
    #initiate the model 
    kmf = KaplanMeierFitter()
    #define the linestyles for TC values 
    tc_styles = {df['ISS_cat'].unique()[0]: '-', #Solid line 
                 df['ISS_cat'].unique()[1]: '--'} #Dotted line
    #Use a colormap for consistent colors per cluster 
    #Specify the unitue clusters  & colormap 
    unique_clusters = sorted(df[cc_Var].unique())
    cmap = plt.cm.get_cmap('Paired', len(unique_clusters))
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    #Plot each cluster
    for cluster in df[cc_Var].unique(): 
        #Iterate through the values of TC
        for tc in df['ISS_cat'].unique(): 
            cluster_tc_data = df[(df[cc_Var] == cluster) & (df['ISS_cat'] == tc)]
            if not cluster_tc_data.empty:             
                kmf.fit(durations=cluster_tc_data['time_to_death'], event_observed=cluster_tc_data['death_occurred'])
                kmf.plot_survival_function(ci_show=True, color=cluster_colors[cluster], linestyle=tc_styles[tc],  label = f'Cluster {cluster} - {tc}')
    #plt.title('Kaplan-Meier Curves by Cluster and Trauma Center Care')
    plt.xlabel('Years After Injury')
    plt.ylim((0,1.0))
    plt.xlim((0,5))
    sort_legend(ax=None, fontsize=fontsize, ncol=ncol) #, loc=loc, bbox_to_anchor=bbox_to_anchor
    plt.ylabel('Alive and at Home Probability')
    plt.tight_layout()
    plt.savefig(f'/filepath/Kaplan_ISS_death{version}.png', dpi=700)
    plt.show()
    
    return 



def sort_legend(ax=None, fontsize=6, ncol=2, loc='best', bbox_to_anchor=None):
    ax = ax or plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles  = zip(*sorted_pairs)
    ax.legend(handles, labels, fontsize=fontsize, ncol=ncol, loc=loc, bbox_to_anchor=bbox_to_anchor)



######################################
#IMPORTING THE CLUSTERED DATA
######################################

## KPROTOYPE FRAILTY OVERALL DATASET
df_frail = pd.read_csv('filepath/filename.csv')

## KPROTOYPE NOT FRAIL OVERALL DATASET
df_NOT_frail = pd.read_csv('/filepath/filename.csv')

#rename clusters 
df_frail['Cluster_Labels'] = df_frail['Cluster_Labels'].replace((0,1),('A','B'))
df_NOT_frail['Cluster_Labels'] = df_NOT_frail['Cluster_Labels'].replace((0,1,2),('C','D','E',))

######################################
#CLEANING THE DATA 
######################################

#Run the clean df function 
df_frail            = clean_data(df_frail)
df_NOT_frail        = clean_data(df_NOT_frail)

#Run the clean df function 
df_frail            = survival_var(df_frail)
df_NOT_frail        = survival_var(df_NOT_frail)

#Create a combined cohort to enable visualizing on same plot
df_combined = pd.concat([df_frail, df_NOT_frail], ignore_index=True, axis=0)



######################################
#RUNNING THE KAPLAN MEIER CURVES
######################################


########  OUTCOME: DEATH  

##  OVERALL KPROTOTYPE CLUSTERING 
kaplan_analysis_death(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_death(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_death(df_combined, 'Cluster_Labels', "combined")

##  STRATIFY BY TRAUMA CENTER ADMISSION
kaplan_analysis_TC_death(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_TC_death(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_TC_death(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)

##  STRATIFY BY SEVERE HEAD INJURY
kaplan_analysis_head_death(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_head_death(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_head_death(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)

##  STRATIFY BY ISS CATEGORY
kaplan_analysis_ISS_death(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_ISS_death(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_ISS_death(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)

##  STRATIFY BY SEX 
kaplan_analysis_sex_death(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_sex_death(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_sex_death(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)



########  OUTCOME: ALIVE AND AT HOME   

##  OVERALL KPROTOTYPE CLUSTERING 
kaplan_analysis(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis(df_combined, 'Cluster_Labels', "combined")

##  STRATIFY BY TRAUMA CENTER ADMISSION
kaplan_analysis_TC(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_TC(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_TC(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)

##  STRATIFY BY SEVERE HEAD INJURY
kaplan_analysis_head(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_head(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_head(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)

##  STRATIFY BY ISS CATEGORY
kaplan_analysis_ISS(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_ISS(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_ISS(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)

##  STRATIFY BY SEX 
kaplan_analysis_sex(df_frail, 'Cluster_Labels', "OG_frail")
kaplan_analysis_sex(df_NOT_frail, 'Cluster_Labels', "OG_NOT_frail")
kaplan_analysis_sex(df_combined, 'Cluster_Labels', "combined", fontsize=6, ncol=2)




######################################
#EVALUATING THE OUTCOME VARIABLE 
######################################

#Creating a varible that have the years until someonf is accepted or admitted to LTC 
df_frail['yrs_A_H'] = (df_frail['event_occurred_dt'] - df_frail['indexdt']).dt.days / 365.25
df_NOT_frail['yrs_A_H'] = (df_NOT_frail['event_occurred_dt'] - df_NOT_frail['indexdt']).dt.days / 365.25


#Frail - Outcomes 1 yr
df_frail_1yr = df_frail[df_frail['time_eligible_yr'] >= 1] 
df_frail_1yr['A_H_yr1']     = df_frail_1yr['yrs_A_H'].apply(lambda x: 'N'if x<1 else 'Yes')
df_frail_1yr['alive_yr1']   = df_frail_1yr['yrs_diff_death'].apply(lambda x: 'N'if x<1 else 'Yes')
outcomes = ["A_H_yr1",  "alive_yr1"]
#df_frail['Cluster_Labels'] = df_frail['Cluster_Labels'].replace((0,1,2,3,4,5,6,7), ('Cluster 0','Cluster 1','Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'))
frail_table1 = TableOne(df_frail_1yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table1.to_csv('/filepath/filename')

#Frail - Outcomes 2 yr
df_frail_2yr = df_frail[df_frail['time_eligible_yr'] >= 2] 
df_frail_2yr['A_H_yr2']     = df_frail_2yr['yrs_A_H'].apply(lambda x: 'N'if x<2 else 'Yes')
df_frail_2yr['alive_yr2']   = df_frail_2yr['yrs_diff_death'].apply(lambda x: 'N'if x<2 else 'Yes')
outcomes = ["A_H_yr2",  "alive_yr2"]
frail_table2 = TableOne(df_frail_2yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table2.to_csv('/filepath/filename.csv')

#Frail - Outcomes 3 yr
df_frail_3yr = df_frail[df_frail['time_eligible_yr'] >= 3] 
df_frail_3yr['A_H_yr3']     = df_frail_3yr['yrs_A_H'].apply(lambda x: 'N'if x<3 else 'Yes')
df_frail_3yr['alive_yr3']   = df_frail_3yr['yrs_diff_death'].apply(lambda x: 'N'if x<3 else 'Yes')
outcomes = ["A_H_yr3",  "alive_yr3"]
frail_table3 = TableOne(df_frail_3yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table3.to_csv('/filepath/filename.csv')

#Frail - Outcomes 4 yr
df_frail_4yr = df_frail[df_frail['time_eligible_yr'] >= 4] 
df_frail_4yr['A_H_yr4']     = df_frail_4yr['yrs_A_H'].apply(lambda x: 'N'if x<4 else 'Yes')
df_frail_4yr['alive_yr4']   = df_frail_4yr['yrs_diff_death'].apply(lambda x: 'N'if x<4 else 'Yes')
outcomes = ["A_H_yr4",  "alive_yr4"]
frail_table4 = TableOne(df_frail_4yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table4.to_csv('/filepath/filename.csv')

#Frail - Outcomes 5 yr
df_frail_5yr = df_frail[df_frail['time_eligible_yr'] >= 5] 
df_frail_5yr['A_H_yr5']     = df_frail_5yr['yrs_A_H'].apply(lambda x: 'N'if x<5 else 'Yes')
df_frail_5yr['alive_yr5']   = df_frail_5yr['yrs_diff_death'].apply(lambda x: 'N'if x<5 else 'Yes')
outcomes = ["A_H_yr5",  "alive_yr5"]
frail_table5 = TableOne(df_frail_5yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table5.to_csv('/filepath/filename.csv')



#Not Frail - Outcomes 1 yr
df_frail_1yr = df_NOT_frail[df_NOT_frail['time_eligible_yr'] >= 1] 
df_frail_1yr['A_H_yr1']     = df_frail_1yr['yrs_A_H'].apply(lambda x: 'N'if x<1 else 'Yes')
df_frail_1yr['alive_yr1']   = df_frail_1yr['yrs_diff_death'].apply(lambda x: 'N'if x<1 else 'Yes')
outcomes = ["A_H_yr1",  "alive_yr1"]
#df_frail['Cluster_Labels'] = df_frail['Cluster_Labels'].replace((0,1,2,3,4,5,6,7), ('Cluster 0','Cluster 1','Cluster 2','Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'))
frail_table1 = TableOne(df_frail_1yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table1.to_csv('/filepath/filename.csv')

#Not Frail - Outcomes 2 yr
df_frail_2yr = df_NOT_frail[df_NOT_frail['time_eligible_yr'] >= 2] 
df_frail_2yr['A_H_yr2']     = df_frail_2yr['yrs_A_H'].apply(lambda x: 'N'if x<2 else 'Yes')
df_frail_2yr['alive_yr2']   = df_frail_2yr['yrs_diff_death'].apply(lambda x: 'N'if x<2 else 'Yes')
outcomes = ["A_H_yr2",  "alive_yr2"]
frail_table2 = TableOne(df_frail_2yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table2.to_csv('/filepath/filename.csv')

#Not Frail - Outcomes 3 yr
df_frail_3yr = df_NOT_frail[df_NOT_frail['time_eligible_yr'] >= 3] 
df_frail_3yr['A_H_yr3']     = df_frail_3yr['yrs_A_H'].apply(lambda x: 'N'if x<3 else 'Yes')
df_frail_3yr['alive_yr3']   = df_frail_3yr['yrs_diff_death'].apply(lambda x: 'N'if x<3 else 'Yes')
outcomes = ["A_H_yr3",  "alive_yr3"]
frail_table3 = TableOne(df_frail_3yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table3.to_csv('/filepath/filename.csv')

#Not Frail - Outcomes 4 yr
df_frail_4yr = df_NOT_frail[df_NOT_frail['time_eligible_yr'] >= 4] 
df_frail_4yr['A_H_yr4']     = df_frail_4yr['yrs_A_H'].apply(lambda x: 'N'if x<4 else 'Yes')
df_frail_4yr['alive_yr4']   = df_frail_4yr['yrs_diff_death'].apply(lambda x: 'N' if x<4 else 'Yes')
outcomes = ["A_H_yr4",  "alive_yr4"]
frail_table4 = TableOne(df_frail_4yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table4.to_csv('/filepath/filename.csv')

#Not Frail - Outcomes 5 yr
df_frail_5yr = df_NOT_frail[df_NOT_frail['time_eligible_yr'] >= 5] 
df_frail_5yr['A_H_yr5']     = df_frail_5yr['yrs_A_H'].apply(lambda x: 'N' if x < 5 else 'Yes')
df_frail_5yr['alive_yr5']   = df_frail_5yr['yrs_diff_death'].apply(lambda x: 'N'if x < 5 else 'Yes')
outcomes = ["A_H_yr5",  "alive_yr5"]
frail_table5 = TableOne(df_frail_5yr, columns=outcomes, missing=False, groupby='Cluster_Labels', pval=True)
frail_table5.to_csv('/filepath/filename.csv')



