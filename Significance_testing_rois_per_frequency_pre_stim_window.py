##Prediction per Event, Feature and ROI, here we plot the pre-stim interval as well (referring to Supplementary Figure S5)
# - performs prediction of informed vs. uninformed decision making
# -  for  each time window seperately
# - from all channels from specific regions of interests (rois) combined 
# - computes whether the prediction roc-AUC is significantly above 0.5 using general linear mixed effects models (This gives warnings as the dataset is relatively small with 6 participants and 10 cross validation iterations). 
# - The results are plotted per event.

## Import
#import some packages and functions from helper_functions.py

import scipy.signal as scisig
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *
from config import path_data

#functions for machine-learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
#functions for statistics
import statsmodels.formula.api as smf
from scipy.stats import norm
#plotting functions
import seaborn as sns
import matplotlib.pyplot as plt

##Settings

#events in the task ("Start Stim" is the start of stimulus presentation,
#"Press" is the time point when the button is pressed)
event_list=["Start Stim","Press"]
#list of the relevant features tested include oscillatory features
#(delta, theta, alpha, beta, gamma) and low-frequency filtered
#time domain signals, corresponding to the respective events
freq_list_rel=[["LF-time","Gamma","Delta"],["Gamma"]]

#list of specific regions of interests corresponding to the respective features and events
#Start Stim: For LF-time those are occipital-temporal gyrus and Hippocampus, for Gamma white-matter and for delta frontal lobe
#Press: For Gamma insula is a relevant roi, also anterior and posterior insula are analyzed seperately
roi_list_rel=[[["oc-temp","Hippocampus"],["white"],["front"]],[["ins","ant_ins","post_ins"]]]

#time windows to be extracted by event to visulize the prediction time course we select a bigger relevant time-window
#time_windows=[[-250,1500],[-1250,250]]
time_windows=[[-1000,1000],[-1000,0]]
#list of participants included in the analysis
participant_list=["P01","P02","P03","P04","P05","P06"]
#path to participant data
path_to_data=str(path_data)


#baseline auc
baseline = 0.5  # chance level

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
    #- list of current event "Start Stim" or "Press"
    event_results_all=[]
    # - list of current oscillatory or time-domain feature
    frequency_results_all=[]
    # - list of current roi
    roi_results_all=[]
    # - list of current time window
    time_window_results_all=[]
    # - list of current participant
    participant_results_all=[]
    # - list of current cross-validation iteration
    cv_number_results_all=[]
    # - list of all auc values
    auc_values_results_all=[]
    # - list of all accuracy values
    acc_values_results_all=[]





    #load the data for every event, frequency and participant
    for i_e, event in enumerate(event_list):
        for i_f,freq in enumerate(freq_list_rel[i_e]):
            for i_r,roi in enumerate(roi_list_rel[i_e][i_f]):
                for i_p,par in enumerate(participant_list):
                    #here the features and corresponding time-window information is loaded
                    path=path_to_data+"/"+par+"/"
                    data=np.load(path+"output/"+par+event+freq+str(time_windows[i_e][0])+"_until_"+str(time_windows[i_e][1])+"features_with_block_stim_trial_num.npz")
                    features=data["features"]
                    time_windows_temp=data["time_array"]
                    #we also load the location data to map channels to rois, this dataframe is computed in "Find anterior channels.py" it also includes a seperation of insular channels into anterior and posterior insula
                    channel_location_struct=pd.read_csv(path+"output/"+'location_struct_insula_labels_'+par+'.csv')
                    #in case the roi is anterior or posterior insula we select insular channels and look up if they belong to anterior or posterior insula
                    if roi=="ant_ins" or roi=="post_ins":
                        loc_temp=channel_location_struct[channel_location_struct["location"].str.lower().str.contains("ins")]
                        if roi=="ant_ins":
                            loc_temp=loc_temp[loc_temp["is_anterior_insula"]==1]
                        elif roi=="post_ins":
                            loc_temp=loc_temp[loc_temp["is_anterior_insula"]==0]
                    #for all other rois we select channels with locations containing the relevant roi-string
                    else:
                        loc_temp=channel_location_struct[channel_location_struct["location"].str.lower().str.contains(roi.lower())]
                    #we select the indices of the channels with this roi
                    ind=loc_temp["Index"]
                    #we iterate over all time-windows
                    for i_t,time in enumerate(time_windows_temp):
                        features_new=copy.deepcopy(features)
                        #we only want to compare informed vs. uninformed decision making so we only include
                        #first and third stimulus presentation and exclude the trials of the second stimulus presentation
                        indices_1_3=my_find(features_new[:,0,-1],'!=',2)[0]
                        mat_class=features_new[indices_1_3,:,:]
                        #remove the last column for the features where label information is saved, features without labels are saved in X
                        temp=mat_class[:,:,:-1]
                        #label information is saved in y
                        y=mat_class[:,0,-1]
                        #we select the channels belonging to the roi based on the computed indices and also select the columns with the relevant time-windows
                        if freq=="LF-time":
                            #for the LF-time feature we downsampled to 8Hz, therefore we need all 4 columns corresponding to the 4 values of the respective 500ms time window
                            temp=temp[:,ind,i_t*4:(i_t*4)+4]
                        else:
                            temp=temp[:,ind,i_t]
                        #we only perform prediciton if the participant has at least one channel of the relevant roi
                        if temp.shape[1]>0:
                            X=temp.reshape(temp.shape[0],-1)
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
                                #fill the previously defined lists with the information of this iteration
                                event_results_all.append(event)
                                frequency_results_all.append(freq)
                                participant_results_all.append(par)
                                time_window_results_all.append(i_t)
                                roi_results_all.append(roi)
                                cv_number_results_all.append(k)
                                auc_values_results_all.append(auc)
                                acc_values_results_all.append(accuracy)
    #we summarize all information in data frames
    #df_prediction_results saves information for every cross-validation iteration, this is important for significance testing
    df_prediction_results=pd.DataFrame({"event":event_results_all,"frequency":frequency_results_all,"participant":participant_results_all,"time":time_window_results_all,"roi":roi_results_all,"cv":cv_number_results_all,"acc":acc_values_results_all,"auc":auc_values_results_all})
    #we also add a column auc_centered that saves the distance of AUC from 0.5 (chance level)
    df_prediction_results['auc_centered'] = df_prediction_results['auc'] - 0.5
    #save the dataframe
    df_prediction_results.to_csv(path_to_data+"/" +roi+"per_event_freq_time_window_prediction_results.csv")  


    ### Statistical testing (event= Start Stim)
    #statistical testing is used to determine if the AUC is significantly above 0.5
   

    #first we do statistics on the Start_Stim features
    df_Start_Stim=df_prediction_results[df_prediction_results["event"]=="Start Stim"]
    # Group by frequency region of interest and time Count total number of tests for Bonferroni adjustment
    grouped_start_stim = df_Start_Stim.groupby(['frequency', 'roi', 'time'])
    n_tests = len(grouped_start_stim)
    alpha_adj = 0.05 / n_tests

    #we apply a linear mixed-effects model to each group
    stats_results_start_stim = grouped_start_stim.apply(lambda g: fit_lmm_stats_one_sided(g, baseline=0.5),include_groups=False).reset_index()
    #we test if the one-sided p-value is below the significance level
    stats_results_start_stim['significant'] = stats_results_start_stim['p_one_sided'] < alpha_adj



     ### Statistical testing (event= Press)
    #second we do statistics on the Button Press features
    df_Press=df_prediction_results[df_prediction_results["event"]=="Press"]
    # Group by frequency region of interest and time Count total number of tests for Bonferroni adjustment
    grouped_press = df_Press.groupby(['frequency', 'roi', 'time'])
    n_tests = len(grouped_press)
    alpha_adj = 0.05 / n_tests
    #we apply a linear mixed-effects model to each group
    stats_results_press = grouped_press.apply(lambda g: fit_lmm_stats_one_sided(g, baseline=0.5),include_groups=False).reset_index()
    #we test if the one-sided p-value is below the significance level
    stats_results_press['significant'] = stats_results_press['p_one_sided'] < alpha_adj


    ##Plotting
    # For each event we plot the results per participant and the distribution (mean and standard deviation across participants). Stars indicate that the AUCs for this feature and event significantly differ from 0.5
  
    ###Plotting (event= Start Stim)
    #First we start again with event "Start Stim"
    relevant_event="Start Stim"
    relevant_event_int=0
    if relevant_event=="Start Stim":
        #We define labels for the relevant time-windows
        time_here=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.10,1.15,1.2,1.25]
        time_here=np.arange(-1 + 0.5/2,
                    1 - 0.5/2 + 0.05,
                    0.05)
        time_labels= [
    f'{round(x, 2):g}' if np.isclose(x / 0.25, round(x / 0.25)) else ''
    for x in time_here
]
        # we only take rows from the prediction and statistics dataframe with the right event
        df_filtered=df_Start_Stim
        stats_results_here=stats_results_start_stim

    # We compute the mean and standard error of the AUC per region of interest and time
    df_summary = df_filtered.groupby(['roi', 'time'])['auc'].agg(['mean', 'sem']).reset_index()

    #We initialize the plot
    fig=plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(111)
    #We define the color palette from the seaborn Dark2 color palette ordered by regions of interest they should be matched to
    palette = sns.color_palette("Dark2")
    palette_new = [palette[1], palette[3], palette[0], palette[2]]
    #we start with the first color value and increase this for each roi
    col_val = 0
    #We iterate over frequency and corresponding roi
    for i_f, freq in enumerate(freq_list_rel[relevant_event_int]):
        for i_r, roi in enumerate(roi_list_rel[relevant_event_int][i_f]):
            #we select the right value
            color = palette_new[col_val]
            #we determine for which time windows the decoding is significant
            sig_points = stats_results_here[
                (stats_results_here["roi"] == roi) &
                (stats_results_here["significant"] ==True)
            ]
            #we add those as x to the plot
            plt.scatter(
                sig_points["time"],
                sig_points["mean_auc"],  # or "mean" depending on what you plot
                color=color,
                marker='x',
                s=200,
                linewidth=6,
                alpha=0.8,
                zorder=20
            )
            #We take all the rows corresponding to the roi
            df_r = df_summary[df_summary['roi'] == roi]
            if df_r.empty:
                continue
            
            
            
            # And plot the mean AUC values for this roi as a lineplot over time
            plt.plot(df_r['time'], df_r['mean'], linewidth=3, label=roi, color=color)
            
            # Around the mean we use the standard error of the mean to add a shaded area in the same color
            plt.fill_between(
                df_r['time'],
                df_r['mean'] - df_r['sem'],
                df_r['mean'] + df_r['sem'],
                color=color,
                alpha=0.2,
                linewidth=0
            )

            #we increase col_val to select the next color for the next roi
            col_val += 1
    #we set the plotting paramters and aesthetics
    plt.tick_params(
        axis='both',       # apply to both x and y axes
        which='both',      # apply to major and minor ticks
        direction='out',   # tick direction
        length=3,          # tick length in points (default ~6)
        width=1,           # tick line width
        labelsize=12       # size of tick labels (can replace fontsize in plt.xticks / plt.yticks)
    )
    for el in [0.5,0.6,0.7,0.8]:

            plt.axhline(el, linestyle='--', color='gray', alpha=0.2)
    for axis in 'left', 'bottom':
        ax.spines[axis].set_linewidth(2)
    sns.despine()
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('AUC',fontsize=20)
    plt.xticks(range(0,len(time_labels)),time_labels,fontsize=15)
    plt.yticks([0.5,0.6,0.7,0.8],["0.5","","0.7",""],fontsize=15)
    #We embed the text in the plots
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42  
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.title('')
    plt.legend().set_visible(False)
    #we save the plot
    fig.savefig(path_to_data+"/plots/prediction_per_frequency_per_roi"+relevant_event+".pdf", dpi=700)
    plt.show()

    ###Plotting (event= Start Stim)
    #second we create a plot for event "Press"
    relevant_event="Press"
    relevant_event_int=1
    if relevant_event=="Press":
        #We define labels for the relevant time-windows
        time_here=[-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0]
        time_labels=["-1","","","","","-0.75","","","","","-0.5","","","","","-0.25","","","","","0"]
        # we only take rows from the prediction and statistics dataframe with the right event
        df_filtered=df_Press
        stats_results_here=stats_results_press

    # We compute the mean and standard error of the AUC per region of interest and time
    df_summary = df_filtered.groupby(['roi', 'time'])['auc'].agg(['mean', 'sem']).reset_index()
    #We initialize the plot
    fig=plt.figure(figsize=(7, 5))
    ax=fig.add_subplot(111)
    #We define the color palette from the seaborn Dark2 color palette ordered by regions of interest they should be matched to
    palette = sns.color_palette("Dark2")
    palette_new = [palette[5], palette[6], palette[7]]
    #we start with the first color value and increase this for each roi
    col_val = 0
    #We iterate over frequency and corresponding roi
    for i_f, freq in enumerate(freq_list_rel[relevant_event_int]):
        for i_r, roi in enumerate(roi_list_rel[relevant_event_int][i_f]):
            #we select the right color value
            color = palette_new[col_val]
            #we determine for which time windows the decoding is significant
            sig_points = stats_results_here[
                (stats_results_here["roi"] == roi) &
                (stats_results_here["significant"] ==True)
            ]
            #we add those as x to the plot
            plt.scatter(
                sig_points["time"],
                sig_points["mean_auc"],  # or "mean" depending on what you plot
                color=color,
                marker='x',
                s=200,
                linewidth=6,
                alpha=0.8,
                zorder=20
            )
            #We take all the rows corresponding to the roi
            df_r = df_summary[df_summary['roi'] == roi]
            if df_r.empty:
                continue
            
            
            
            # And plot the mean AUC values for this roi as a lineplot over time
            plt.plot(df_r['time'], df_r['mean'], linewidth=3, label=roi, color=color)
            
            # Around the mean we use the standard error of the mean to add a shaded area in the same color
            plt.fill_between(
                df_r['time'],
                df_r['mean'] - df_r['sem'],
                df_r['mean'] + df_r['sem'],
                color=color,
                alpha=0.2,
                linewidth=0
            )
            #we increase col_val to select the next color for the next roi
            col_val += 1
    #we set the plotting paramters and aesthetics
    plt.tick_params(
        axis='both',       # apply to both x and y axes
        which='both',      # apply to major and minor ticks
        direction='out',   # tick direction
        length=3,          # tick length in points (default ~6)
        width=1,           # tick line width
        labelsize=12       # size of tick labels (can replace fontsize in plt.xticks / plt.yticks)
    )
    for el in [0.5,0.6,0.7,0.8,0.9]:

            plt.axhline(el, linestyle='--', color='gray', alpha=0.2)
    for axis in 'left', 'bottom':
        ax.spines[axis].set_linewidth(2)
    sns.despine()
    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('AUC',fontsize=22)
    plt.xticks(range(0,len(time_labels)),time_labels,fontsize=19)
    plt.yticks([0.5,0.6,0.7,0.8,0.9],["0.5","","0.7","","0.9"],fontsize=19)
    plt.title('')
    plt.legend().set_visible(False)
    plt.ylim(0.4,0.9)
    #We embed the text in the plots
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42  
    plt.rcParams['font.family'] = 'DejaVu Sans'
    #we save the plot
    if relevant_event=="Start Stim":
        fig.savefig(path_to_data+"/plots/prediction_per_frequency_per_roi"+relevant_event+"prestim_interval.pdf", dpi=700)
    plt.show()

