# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:38:39 2021

@author: panay
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
pd.options.plotting.backend = "plotly"

def Separate(df):
    #separate the 2 people 
    df1= df.loc[df[speaker] == df[speaker][0]]
    df2= df.loc[df[speaker] != df1[speaker][0]]
    
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    
    Person_1 = df1[speaker][0]
    Person_2 = df2[speaker][0]
    
    return df,Person_1,Person_2

def Wordcount(df,Person_1,Person_2):
    List = []
    #count words
    for i in range(len(df.index)):
        NumWords=0
        sentence=df[message][i]
        #checks if the sentence is NaN
        if isinstance(sentence, str):
            NumWords =NumWords + len(sentence.split())
        else:
            NumWords=0
            
        List.append(NumWords)
        
    df['Num_Words']=List
    #separate the words for each speaker
    df['Num_Words_1']=df['Num_Words'].loc[df[speaker]==Person_1]
    df['Num_Words_2']=df['Num_Words'].loc[df[speaker]==Person_2] 
    df.drop('Num_Words',inplace=True,axis=1)
    df=df.fillna(0)
    
    return df


def Message_Spam(df,Person_1,Person_2):
    
    list_1 = []
    list_2 = []
    count_1 = 0
    count_2 = 0
    for i in range(len(df)):
        
        
        if df[speaker][i] == Person_1:
            count_1 +=1
            count_2 = 0
        elif df[speaker][i] == Person_2:
            count_2 +=1  
            count_1 = 0
        
        list_1.append(count_1)
        list_2.append(count_2)
            
    df['Spam_1'] = list_1
    df['Spam_2'] = list_2
    
    return df

def CountSpamWords(df):
    NumWords_1 = 0
    NumWords_2 = 0
    NumWordsSpam_1 = []
    NumWordsSpam_2 = []
    for i in range(len(df)):
        if df['Spam_1'][i] != 0:
            NumWords_1 += df['Num_Words_1'][i]
            NumWords_2 = 0
        elif df['Spam_2'][i] != 0:
            NumWords_2 += df['Num_Words_2'][i]
            NumWords_1 = 0
            
        NumWordsSpam_1.append(NumWords_1)
        NumWordsSpam_2.append(NumWords_2)
        
    df['Number_of_Continuous_Words_1'] = NumWordsSpam_1
    df['Number_of_Continuous_Words_2'] = NumWordsSpam_2
     
    return df

def Density(df):
    
    df['Dn_1']= df['Number_of_Continuous_Words_1'].div(df['Spam_1'])
    df['Dn_2']= df['Number_of_Continuous_Words_2'].div(df['Spam_2'])
    df=df.fillna(0)
    
    return df

def Time(df,name):
    
    dfTime= df.groupby(['time',speaker]).sum()
    dfTime=dfTime.reset_index()
    dfTime=dfTime.drop(speaker,axis=1)
    dfTime = dfTime.set_index('time')
    
    fig = dfTime.boxplot(title="Person 1/ Person 2", template="simple_white")
    fig.show()
    fig.write_html("./"+str(name)+'_'+'Time_box'+ ".html")
    
    fig = px.bar(dfTime,title="Person 1/ Person 2", template="simple_white",
                      labels=dict(index="Index of Message", value="NumWords", variable="option"))
    fig.show()
    fig.write_html("./"+str(name)+'_'+'Time'+ ".html")
    
    return dfTime

def Avg(df,filename,GlobalMetrics):
    #get mean Global metrics
    avg=df.mean(axis=0)
    columns_to_use=avg.index.tolist()
    avg=list(avg)
    #Save them according to their file
    GlobalMetrics[str(filename)]=avg
    dfGlobal=pd.DataFrame.from_dict(GlobalMetrics,orient='index',columns=columns_to_use )
    
    dfGlobal.to_csv('./Data/Mean_GlobalMetrics.csv')

def Final_Split(df):
     #separate the 2 people 
    df1= df.loc[df[speaker] == df[speaker][0]]
    
    df2= df.loc[df[speaker] != df1[speaker][0]]
    
    
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    
    df1.drop('Num_Words_2',inplace=True,axis=1)
    df2.drop('Num_Words_1',inplace=True,axis=1)
    df1.drop('Spam_2',inplace=True,axis=1)
    df2.drop('Spam_1',inplace=True,axis=1)
    df1.drop('Number_of_Continuous_Words_2',inplace=True,axis=1)
    df2.drop('Number_of_Continuous_Words_1',inplace=True,axis=1)
    df1.drop('Dn_2',inplace=True,axis=1)
    df2.drop('Dn_1',inplace=True,axis=1)

    return df1,df2
    
 
    
def plotting(df,name,Typeplot):
    
    if Typeplot == 'Num_Words':
        fields_to_plot=['Num_Words_1','Num_Words_2','Number_of_Continuous_Words_1','Number_of_Continuous_Words_2']
        dfplot=df[fields_to_plot]
        
        fig = px.bar(dfplot,title="Person 1/ Person 2", template="simple_white",
                      labels=dict(index="Index of Message", value="NumWords", variable="option"))
        fig.show()
        fig.write_html("./"+str(name)+'_'+str(Typeplot)+ ".html")
    
    elif Typeplot == 'Boxplot':
        fields_to_plot=['Num_Words_1','Num_Words_2','Number_of_Continuous_Words_1','Number_of_Continuous_Words_2','Spam_1','Spam_2','Dn_1','Dn_2']
        dfplot=df[fields_to_plot]
        
        fig = dfplot.boxplot(title="Person 1/ Person 2", template="simple_white")
        fig.show()
        fig.write_html("./"+str(name)+'_'+str(Typeplot)+ ".html")
    
    elif Typeplot == 'Spam':
        fields_to_plot=['Spam_1','Spam_2']
        dfplot=df[fields_to_plot]
        
        fig = px.line(dfplot,title="Person 1/ Person 2", template="simple_white",
                      labels=dict(index="Index of Message", value="Num of Messages", variable="option"))
        fig.show()
        fig.write_html("./"+str(name)+'_'+str(Typeplot)+ ".html")
    
    elif Typeplot == 'Dn':
        fields_to_plot=['Dn_1','Dn_2']
        dfplot=df[fields_to_plot]
        
        fig = px.bar(dfplot,title="Person 1/ Person 2", template="simple_white",
                      labels=dict(index="Index of Message", value="Density", variable="option"))
        fig.show()
        fig.write_html("./"+str(name)+'_'+str(Typeplot)+ ".html")
        

if __name__ == '__main__':
    import pandas as pd
    import os
    from glob import glob
    PATH = "./Data"
    EXT = "*.csv"
    
    GlobalMetrics={}
    
    
    all_csv_files = [file 
                      for path, subdir, files in os.walk(PATH)
                      for file in glob(os.path.join(path, EXT))]
    
    for csv_file in all_csv_files:
        df = pd.read_csv(csv_file) #, parse_dates=['Time'])
        
        #consider pan-12 and perverted justice
        if df.columns[0] == 'speaker_role':
            #columns to use
            speaker='speaker_role'
            message='message'
            do_time=False
        else:
            speaker='speaker'
            message='content'
            columns=[speaker,message,'time']
            #read the csv with the time column
            df = pd.read_csv(csv_file, usecols=columns)
            do_time=True
        
            
        # functions used for all the csvs
        df,Person_1,Person_2 = Separate(df)    
        
        
        df = Wordcount(df,Person_1,Person_2)
         
        
        df= Message_Spam(df,Person_1,Person_2)
        
        df = CountSpamWords(df)
        
        df = Density(df)
        #split datasets to view in variable explorers
        df1,df2 = Final_Split(df)
        
        

        #create files and save dataframe with metrics and its plots
        base=os.path.basename(csv_file)
        file_name=os.path.splitext(csv_file)[0]
        
        #remove extension to create folder
        os.mkdir(file_name)
        
        #Create the global metrics dictionary
        Avg(df,base,GlobalMetrics)
        
        #save file csv in the new directory 
        df.to_csv('./'+file_name+'/'+base +'.csv',index=False)
        
        #name to save plots in the correct folder
        name=file_name+'/'+base 
        Typeplot=['Num_Words','Boxplot','Spam','Dn']
        for plotType in Typeplot:
            plotting(df,name,plotType)
        
        if do_time:
            dfTime = Time(df,name) 
        dfTime.to_csv('./'+file_name+'/'+base +'_Time.csv',index=False)    
        
   ###### delete the csvs that was analysed ###### 
        # os.remove(csv_file)
      
