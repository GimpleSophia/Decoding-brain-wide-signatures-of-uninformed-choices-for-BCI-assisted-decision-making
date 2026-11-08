##Prediction per Event and Features, determine which block (presentation 1,2 or 3) to leave out by setting trial_number_to_leave_out, run for each block as lave out to plop_by_leave_repetition_out
# - performs prediction of informed vs. uninformed decision making
# - for all oscillatory or time domain features all channels and time-windows are combined
# -computes whether the prediction roc-AUC is significantly above 0.5 using general linear mixed effects models (This gives warnings as the dataset is relatively small with 6 participants and 10 cross validation iterations). 
# - The results are plotted per event.
##Import 
#import some packages and functions from helper_functions.py

import scipy.signal as scisig
import pandas as pd
from helper_functions import *
from config import path_data

#plotting functions
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#functions for machine-learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
#functions for statistics
import statsmodels.formula.api as smf

##Settings

#events in the task ("Start Stim" is the start of stimulus presentation,
#"Press" is the time point when the button is pressed)
event_list=["Start Stim","Press"]
#event_list=["Start Stim"]
#list of the features tested include oscillatory features
#(delta, theta, alpha, beta, gamma) and low-frequency filtered
#time domain signals
freq_list=["Delta","Theta","Alpha","Beta","Gamma","LF-time"]
#freq_list=["LF-time"]
#time windows to be extracted by event here we use 1 sencond after stimulus onset and 1 second
#before button press 
time_windows=[[0,1000],[-1000,0]]
#list of participants included in the analysis
participant_list=["P01","P02","P03","P04","P05","P06"]
#path to participant data
path_to_data=str(path_data)
#determine which block to leave out
trial_number_to_leave_out=3

##Define machine learning pipeline
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
        for i_f,freq in enumerate(freq_list):
            for i_p,par in enumerate(participant_list):
                #here the features and corresponding time-window information is loaded
                path=path_to_data+"/"+par+"/"
                data=np.load(path+"output/"+par+event+freq+str(time_windows[i_e][0])+"_until_"+str(time_windows[i_e][1])+"features_with_block_stim_trial_num.npz")
                features=data["features"]
                time_windows_temp=data["time_array"]
                block_arr=data["block_numbers"]
                not_min_1=np.where(block_arr!=-1)[0]
                block_arr=block_arr[not_min_1]
                features_new=copy.deepcopy(features)
                #we only want to compare informed vs. uninformed decision making so we only include
                #first and third stimulus presentation and exclude the trials of the second stimulus presentation
                indices_1_3=my_find(features_new[:,0,-1],'!=',trial_number_to_leave_out)[0]
                mat_class=features_new[indices_1_3,:,:]
                block_arr=block_arr[indices_1_3]
                #remove the last column for the features where label information is saved, features without labels are saved in X
                temp=mat_class[:,:,:-1]
                X=temp.reshape(temp.shape[0],-1)
                #label information is saved in y
                y=mat_class[:,0,-1]
                #crossvalidation
                for ind in range(1,max(block_arr)+1):
                    #current train and test dataset
                    
                    test_indices=np.where(block_arr==ind)[0]
                    #print(test_indices)
                    train_indices=np.where(block_arr!=ind)[0]
                    X_train=X[train_indices,:]
                    y_train=y[train_indices]
                    X_test=X[test_indices,:]
                    y_test=y[test_indices]
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
                    cv_number_results_all.append(ind)
                    auc_values_results_all.append(auc)
                    acc_values_results_all.append(accuracy)
    #we summarize all information in data frames
    #df_prediction_results saves information for every cross-validation iteration, this is important for significance testing
    df_prediction_results=pd.DataFrame({"event":event_results_all,"frequency":frequency_results_all,"participant":participant_results_all,"cv":cv_number_results_all,"acc":acc_values_results_all,"auc":auc_values_results_all})
    #we also add a column auc_centered that saves the distance of AUC from 0.5 (chance level)
    df_prediction_results['auc_centered'] = df_prediction_results['auc'] - 0.5
    #save the dataframe
    df_prediction_results.to_csv(path_to_data+"/per_event_freq_all_time_window_prediction_results_one_block_out"+str(trial_number_to_leave_out)+".csv")  

    ### Statistical testing
    #statistical testing is used to determine if the AUC is significantly above 0.5
    
    #again we want to summarize the results in a dataframe thus we initialize some arrays
    # - current events
    event_list_stats=[]
    # - current frequency
    freq_list_stats=[]
    # - corresponding estimate
    b_list_stats=[]
    # - corresponding standard error
    se_list_stats=[]
    # - corresponding z-value
    z_list_stats=[]
    # - corresponding p-value
    p_list_stats=[]
    # - corresponding one-sided p-value
    p_one_sided_stats=[]
    # - corresponding 95% confidence interval
    ci_list_stats=[]
    # - corresponding mean AUC
    auc_mean_stats=[]
    # - corresponding standard deviation of AUC
    auc_std_stats=[]
    # - corrsponding list of AUC per participant
    auc_participant_means_stats=[]

    #we iterate over events
    for i_e, event in enumerate(event_list):
        #take only rows corresponding to this event
        df_prediction_event=df_prediction_results[df_prediction_results["event"]==event]
        #iterate over features
        for i_f,freq in enumerate(freq_list):
            #take only rows corresponding to this feature
            df_prediction_freq=df_prediction_event[df_prediction_event["frequency"]==freq]
            #define the mixed linear model
            model = smf.mixedlm("auc_centered ~ 1", df_prediction_freq, groups=df_prediction_freq["participant"])
            model = smf.mixedlm(
            "auc_centered ~ 1",
            df_prediction_freq,
            groups=df_prediction_freq["participant"]
            )
            #fit model
            result = model.fit(method="nm")
            print(freq, trial_number_to_leave_out,result.summary())
            #we verify that the model converged
            if result.converged:
                print("Model converged successfully")
            else:
                print("Model did NOT converge")
            #extract the relevant model parameters
            params = result.params["Intercept"]               # Coefficients (B)
            se = result.bse["Intercept"]                       # Standard errors
            z = result.tvalues["Intercept"]                     # z/t statistics
            p = result.pvalues["Intercept"]   
            p_one_sided=1-norm.cdf(z)                   # p-value
            ci = result.conf_int().loc["Intercept"] 
            #save all computed parameters with corresponding event, feature and auc means                # Confidence intervals
            event_list_stats.append(event)
            freq_list_stats.append(freq)
            b_list_stats.append(params)
            se_list_stats.append(se)
            z_list_stats.append(z)
            p_list_stats.append(p)
            p_one_sided_stats.append(p_one_sided)
            ci_list_stats.append(ci)
            participant_auc_means=[]
            #also extract auc for each participant individually and save it as a list
            for i_p, par in enumerate(participant_list):
                df_prediction_par=df_prediction_freq[df_prediction_freq["participant"]==par]
                participant_auc_means.append(df_prediction_par["auc"].mean())
            auc_mean_stats.append(np.mean(participant_auc_means))
            auc_std_stats.append(np.std(participant_auc_means))
            auc_participant_means_stats.append(participant_auc_means)
    #save the statistics in a dataframe and save it
    df_stats_results = pd.DataFrame({"event":event_list_stats,"frequency":freq_list_stats,"b":b_list_stats,"se":se_list_stats,"z":z_list_stats,"p":p_list_stats,"p_one_sided":p_one_sided_stats,"CI":ci_list_stats,"auc_mean":auc_mean_stats,"auc_std:":auc_std_stats,"auc_participant_means":auc_participant_means_stats})
    df_stats_results.to_csv(path_to_data+"/per_event_freq_all_time_window_statistic_results_LOBO_"+str(trial_number_to_leave_out)+".csv")  





    ### Plotting (event =Start Stim)
    # For event Start Stim we plot the results per participant and the distribution (mean and standard deviation across participants). Stars indicate that the AUCs for this feature and event significantly differ from 0.5
 

    # Event of interest select 'Start Stim'
    #event_of_interest = 'Press'
    event_of_interest="Start Stim"
    # we only take rows from the prediction and statistics dataframe with the right event
    df_filtered = df_prediction_results[df_prediction_results['event'] == event_of_interest]
    df_filtered_stats=df_stats_results[df_stats_results['event'] == event_of_interest]

    #we compute a list with 0 or 1 values which indicate whether the corresponding feature has a one-sided p-value
    #lower than the bonferroni corrected significance level
    is_sig_list=[]
    for i_f, freq in enumerate(freq_list):
        df_stats_temp=df_filtered_stats[df_filtered_stats["frequency"]==freq]
        p_val=df_stats_temp["p_one_sided"]
        is_sig=int(p_val.iloc[0]<0.05/len(freq_list))
        is_sig_list.append(is_sig)

    # We compute the mean AUC per participant × frequency
    df_means = df_filtered.groupby(['frequency', 'participant'], as_index=False)['auc'].mean()

    # We ensure frequencies are ordered according to freq_list
    df_means['frequency'] = pd.Categorical(df_means['frequency'], categories=freq_list, ordered=True)
    df_means = df_means.sort_values('frequency')

    # We c the ompute group mean and SD per frequency
    summary = df_means.groupby('frequency',observed=False)['auc'].agg(['mean', 'std']).reindex(freq_list).reset_index()
    print(summary)
    # We map participants to colors from the seaborn tab10 color palette
    participant_order = sorted(df_means['participant'].unique())  # or your own list
    palette = dict(zip(participant_order, sns.color_palette('tab10', n_colors=len(participant_order))))

    # We initialize the plot
    fig=plt.figure(figsize=(7, 6))
    ax = plt.gca()
    sns.set_theme(style="ticks")
    x_positions = np.arange(len(freq_list))

    # We plot a grid indicating AUCs from 0.4 to 1
    for y in [0.4, 0.6, 0.7, 0.8, 0.9,1]:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8,zorder=1)
    #We draw errorboxes that show the standard deviation around the mean over all participants
    for x, (_, row) in zip(x_positions, summary.iterrows()):
        mean, std = row['mean'], row['std']
       
        if pd.isna(mean):
            continue

        # We draw the errorboxes as a rectangle (mean ± SD)
        rect = Rectangle(
            (x - 0.25, mean - std),
            width=0.5,
            height=2 * std,
            facecolor='white',
            edgecolor='black',
            alpha=1,zorder=5
        )
        ax.add_patch(rect)

        # We draw the mean lines
        plt.plot([x - 0.25, x + 0.25], [mean, mean], color='black', linewidth=2,zorder=10)

    # We plot AUC for each participant as scatterpoints
    jitter_strength = 0.08  # adjust as needed

    for i, freq in enumerate(freq_list):
        df_freq = df_means[df_means['frequency'] == freq]

        # Count occurrences of each AUC
        auc_counts = df_freq['auc'].value_counts()
        count=0
        for _, row in df_freq.iterrows():
            auc_value = row['auc']
            
            # Only add scatter if this AUC occurs more than once
            if auc_counts[auc_value] > 1:
                jitter = 0.03
                if count==0:
                    pos=i-jitter
                else:
                    pos=i+jitter
                count=1
            else:
                pos=i
            plt.scatter(
                    pos,
                    auc_value,
                    color=palette[row['participant']],
                    s=60,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=10
                )


    #We use stars to indicate if a features AUCs are significantly different from 0.5
    y_star = 0.95  # height to place the star
    for i, sig in enumerate(is_sig_list):
        if sig == 1:
            plt.scatter(i, y_star, marker="*",color="black",s=70)
    #we add a line at 0.5 to indicate the chance level
    plt.axhline(y=0.5, color='gray', linestyle='-', linewidth=2,zorder=12)


    # We customize the axes and aesthetics
    plt.xticks(x_positions, freq_list, rotation=45,fontsize=20)
    plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],["","","0.5","","0.7","","0.9",""],fontsize=20)
    plt.ylabel('Mean AUC',fontsize=20)
    plt.ylim(0.3, 1)

    plt.tick_params(
        axis='both',
        which='both',
        direction='out',
        length=2,
        width=1,
        bottom=True,
        left=True,
        top=False,
        right=False
    )

    sns.despine()
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(2.5)
    plt.tight_layout()
    #embed the text in the figure
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42  
    plt.rcParams['font.family'] = 'DejaVu Sans'
    #save the figure
    #fig.savefig(path_to_data+"/plots/prediction_per_frequency_all_times_"+event_of_interest+"one_block_out.pdf", dpi=700)
    plt.show()



    ### Plotting (event =Press)
    #For event Press we plot the results per participant and the distribution (mean and standard deviation across participants). Stars indicate that the AUCs for this feature and event significantly differ from 0.5


    # Event of interest select 'Press'
    event_of_interest = 'Press'

    # we only take rows from the prediction and statistics dataframe with the right event
    df_filtered = df_prediction_results[df_prediction_results['event'] == event_of_interest]
    df_filtered_stats=df_stats_results[df_stats_results['event'] == event_of_interest]

    #we compute a list with 0 or 1 values which indicate whether the corresponding feature has a one-sided p-value
    #lower than the bonferroni corrected significance level
    is_sig_list=[]
    for i_f, freq in enumerate(freq_list):
        df_stats_temp=df_filtered_stats[df_filtered_stats["frequency"]==freq]
        p_val=df_stats_temp["p_one_sided"]
        is_sig=int(p_val.iloc[0]<0.05/len(freq_list))
        is_sig_list.append(is_sig)

    # We compute the mean AUC per participant × frequency
    df_means = df_filtered.groupby(['frequency', 'participant'], as_index=False)['auc'].mean()

    # We ensure frequencies are ordered according to freq_list
    df_means['frequency'] = pd.Categorical(df_means['frequency'], categories=freq_list, ordered=True)
    df_means = df_means.sort_values('frequency')

    # We c the ompute group mean and SD per frequency
    summary = df_means.groupby('frequency',observed=False)['auc'].agg(['mean', 'std']).reindex(freq_list).reset_index()
    print(summary)
    # We map participants to colors from the seaborn tab10 color palette
    participant_order = sorted(df_means['participant'].unique())  # or your own list
    palette = dict(zip(participant_order, sns.color_palette('tab10', n_colors=len(participant_order))))
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="ticks")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    # We initialize the plot
    fig=plt.figure(figsize=(7, 6))
    ax = plt.gca()
    sns.set_theme(style="ticks")
    x_positions = np.arange(len(freq_list))

    # We plot a grid indicating AUCs from 0.4 to 1
    for y in [0.4, 0.6, 0.7, 0.8, 0.9,1]:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8,zorder=1)
    #We draw errorboxes that show the standard deviation around the mean over all participants
    for x, (_, row) in zip(x_positions, summary.iterrows()):
        mean, std = row['mean'], row['std']
        
        if pd.isna(mean):
            continue

        # We draw the errorboxes as a rectangle (mean ± SD)
        rect = Rectangle(
            (x - 0.25, mean - std),
            width=0.5,
            height=2 * std,
            facecolor='white',
            edgecolor='black',
            alpha=1,zorder=5
        )
        ax.add_patch(rect)

        # We draw the mean lines
        plt.plot([x - 0.25, x + 0.25], [mean, mean], color='black', linewidth=2,zorder=10)

    # We plot AUC for each participant as scatterpoints
    jitter_strength = 0.08  # adjust as needed

    for i, freq in enumerate(freq_list):
        df_freq = df_means[df_means['frequency'] == freq]

        # Count occurrences of each AUC
        auc_counts = df_freq['auc'].value_counts()
        count=0
        for _, row in df_freq.iterrows():
            auc_value = row['auc']
            
            # Only add scatter if this AUC occurs more than once
            if auc_counts[auc_value] > 1:
                jitter = 0.03
                if count==0:
                    pos=i-jitter
                else:
                    pos=i+jitter
                count=1
            else:
                pos=i
            plt.scatter(
                    pos,
                    auc_value,
                    color=palette[row['participant']],
                    s=60,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=10
                )


    #We use stars to indicate if a features AUCs are significantly different from 0.5
    y_star = 0.95  # height to place the star
    for i, sig in enumerate(is_sig_list):
        if sig == 1:
            plt.scatter(i, y_star, marker="*",color="black",s=70)
    #we add a line at 0.5 to indicate the chance level
    plt.axhline(y=0.5, color='gray', linestyle='-', linewidth=2,zorder=12)


    # We customize the axes and aesthetics
    plt.xticks(x_positions, freq_list, rotation=45,fontsize=20)
    plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],["","","0.5","","0.7","","0.9",""],fontsize=19)
    plt.ylabel('Mean AUC',fontsize=20)
    plt.ylim(0.3, 1)

    plt.tick_params(
        axis='both',
        which='both',
        direction='out',
        length=2,
        width=1,
        bottom=True,
        left=True,
        top=False,
        right=False
    )

    sns.despine()
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(2.5)
    plt.tight_layout()
    #embed the text in the figure
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42  
    plt.rcParams['font.family'] = 'DejaVu Sans'
    #save the figure
    #fig.savefig(path_to_data+"/plots/prediction_per_frequency_all_times_"+event_of_interest+"_one_block_out.pdf", dpi=700)
    plt.show()