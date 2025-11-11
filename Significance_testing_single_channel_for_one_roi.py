##Single channel prediction for insular channels
# - performs prediction of informed vs. uninformed decision making
# -  for gamma (or other specified) features
# - from all time windows combined
# - for single insula (also other roi possible) channels per participant
# - computes a ttest to determine whether the prediction AUC is significantly above 0.5. 
# - t-values per channel are plotted in skript Plot_insula_single_channel_t_values.py
## Import 
#import some packages and functions from helper_functions.py

import scipy.signal as scisig
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *

#plotting functions
import seaborn as sns
import matplotlib.pyplot as plt

#functions for machine-learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
#functions for statistics
import statistics as stats
import statsmodels.formula.api as smf
import scipy
from config import path_data

## Settings

#events in the task ("Start Stim" is the start of stimulus presentation,
#"Press" is the time point when the button is pressed)
event_list=["Press"]
#list of the features tested include oscillatory features
#(delta, theta, alpha, beta, gamma) and low-frequency filtered
#time domain signals saved in lists corresponding to the event
freq_list_rel=[["Gamma"]]
#list of specific regions of interests corresponding to the respective features and events
roi_list_rel=[[["ins"]]]
#time windows to be extracted by event here we use 1 sencond after stimulus onset and 1 second
#before button press 
time_windows=[[-1000,0]]
#list of participants included in the analysis
participant_list=["P01","P02","P03","P04","P05","P06"]
#path to participant data
path_to_data=str(path_data)

#define machine learning pipeline
#we use 10-fold cross-validation
cv = StratifiedKFold(10, shuffle=False)
#we normalize the features using a minmax scaler
scal=MinMaxScaler()
#our classifier is a logistic regression classifier
clf=LogisticRegression(random_state=0)
#we define a pipeline that combines the scaler and the classifier
grid = Pipeline([('scal',scal),('clf', clf)])

if __name__=='__main__':   
                                 
    #We want to save all classification in one dataframe for this we initialize some arrays to intermediately save the data
    # - list of current event "Start Stim" or "Press"
    event_results_all=[]
    # - list of current oscillatory or time-domain feature
    frequency_results_all=[]
    # - list of current roi
    roi_results_all=[]
    # - list of current channel
    channel_results_all=[]
    # - list of the location label of the current channel
    location_results_all=[]
    # - list of current participant
    participant_results_all=[]
    # - list of current cross-validation iteration
    cv_number_results_all=[]
    # - list of all auc values
    auc_values_results_all=[]
    # - list of all accuracy values
    acc_values_results_all=[]
    # - list of all t-values
    tval_results_all=[]
    # - list of all p-values
    pval_results_all=[]
    # - list indicating if the current channel is anterior (1) or posterior (0)
    isanterior_results_all=[]
    # - warped x-coordinates in MNI space
    MNI_coord_x_results_all=[]
    # - warped y-coordinates in MNI space
    MNI_coord_y_results_all=[]
    # - warped z-coordinates in MNI space
    MNI_coord_z_results_all=[]
    # - Name of the current channel
    channel_name_results_all=[]



    #load the data for every event, frequency, roi and participant
    for i_e, event in enumerate(event_list):
        for i_f,freq in enumerate(freq_list_rel[i_e]):
            for i_r,roi in enumerate(roi_list_rel[i_e][i_f]):
                for i_p,par in enumerate(participant_list):
                    #here the features and corresponding time-window information is loaded
                    path=path_to_data+"/"+par+"/"
                    data=np.load(path+"output/"+par+event+freq+str(time_windows[i_e][0])+"_until_"+str(time_windows[i_e][1])+"features.npz")
                    features=data["features"]
                    time_windows_here=data["time_array"]
                    #we also load the location data to map channels to rois, this dataframe is computed in "Find anterior channels.py" it also includes a seperation of insular channels into anterior and posterior insula
                    channel_location_struct=pd.read_csv(path+"output/"+'location_struct_insula_labels_'+par+'.csv')
                    #we take only rows corresponding to the right roi
                    loc_temp=channel_location_struct[channel_location_struct["location"].str.lower().str.contains(roi.lower())]
                    #we iterate over single channels (=rows in the location dataframe)
                    for i_c,row in loc_temp.iterrows():
                        #we get the index of the current channel
                        ind=row["Index"]
                        features_new=copy.deepcopy(features)
                        #we only want to compare informed vs. uninformed decision making so we only include
                        #first and third stimulus presentation and exclude the trials of the second stimulus presentation
                        indices_1_3=my_find(features_new[:,0,-1],'!=',2)[0]
                        mat_class=features_new[indices_1_3,:,:]
                        #remove the last column for the features where label information is saved, features without labels are saved in X
                        temp=mat_class[:,:,:-1]
                        #we select only the current channel
                        temp=temp[:,ind,:]

                        X=temp.reshape(temp.shape[0],-1)

                        #label information is saved in y
                        y=mat_class[:,0,-1]
                        all_acc=[]
                        all_auc=[]
                        #crossvalidation
                        for k,(train, test) in enumerate(cv.split(np.array(X),np.array(y))):
                                #current train and test dataset
                                X_train=X[train,:]
                                y_train=y[train]
                                X_test=X[test,:]
                                y_test=y[test]
                                #train the classifier
                                grid.fit(X_train, y_train)
                                #predict labels and compute prediction accuracy
                                y_pred=grid.predict(X_test)
                                accuracy = accuracy_score(y_pred, y_test)
                                #predict label probability and compute roc AUC scores
                                predicted_probs = grid.predict_proba(X_test)[:,1]
                                auc=roc_auc_score(y_test,predicted_probs)
                                #append accuracy and auc to a per-channel list
                                all_acc.append(accuracy)
                                all_auc.append(auc)
                        #fill the previously defined lists with the information of this iteration        
                        event_results_all.append(event)
                        frequency_results_all.append(freq)
                        participant_results_all.append(par)
                        roi_results_all.append(roi)
                        #perform statistical testing to test whether the auc values are significantly above 0.5 and also append the result
                        t_stat, p = scipy.stats.ttest_1samp(all_auc, popmean=0.5)
                        auc_values_results_all.append(np.mean(all_auc))
                        acc_values_results_all.append(np.mean(all_acc))
                        tval_results_all.append(t_stat)
                        pval_results_all.append(p)
                        channel_name_results_all.append(row["channel_name"])
                        location_results_all.append(row["location"])
                        isanterior_results_all.append(row["is_anterior_insula"])
                        MNI_coord_x_results_all.append(row["x_MNI"])
                        MNI_coord_y_results_all.append(row["y_MNI"])
                        MNI_coord_z_results_all.append(row["z_MNI"])
    #we summarize all information in data frames
    #df_prediction_results saves information for every channel
    df_prediction_results=pd.DataFrame({"event":event_results_all,"frequency":frequency_results_all,"participant":participant_results_all,"roi":roi_results_all,"channel_name":channel_name_results_all,"location":location_results_all,"is_anterior":isanterior_results_all,"acc":acc_values_results_all,"auc":auc_values_results_all,"t_val": tval_results_all,"pval":pval_results_all,"MNI_x":MNI_coord_x_results_all,"MNI_y":MNI_coord_y_results_all,"MNI_z":MNI_coord_z_results_all})
    #we also add a column auc_centered that saves the distance of AUC from 0.5 (chance level)
    df_prediction_results['auc_centered'] = df_prediction_results['auc'] - 0.5
    #save the dataframe
    df_prediction_results.to_csv(path_to_data+"/single_channel"+event+freq+"prediction_results.csv")  