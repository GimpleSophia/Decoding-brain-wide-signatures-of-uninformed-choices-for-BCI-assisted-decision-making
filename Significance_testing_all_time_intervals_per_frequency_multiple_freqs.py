##Prediction per Event and Features, combining multiple features, see supplementary materials S4
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
from scipy.stats import ttest_rel

##Settings

#events in the task ("Start Stim" is the start of stimulus presentation,
#"Press" is the time point when the button is pressed)
event_list=["Start Stim","Press"]
#list of the features tested include oscillatory features
#(delta, theta, alpha, beta, gamma) and low-frequency filtered
#time domain signals
#freq_list=["Delta","Theta","Alpha","Beta","Gamma","LF-time"]
#here we define the primary frequency for event "Start Stim" this is LF-time  and for "Press"
#that is gamma, and then we combine it with the other frequencies from freq_list
#freq_primary=["Gamma"]
freq_primary=["LF-time"]
freq_list=["Delta","Theta","Alpha","Beta","Gamma","LF-time"]


plot_freqs = [
        "GammaGamma",
        "GammaAlpha",
        "GammaDelta",
        "GammaLF-time"
    ]

plot_labels = [
        "Gamma",
        "+Alpha",
        "+Delta",
        "+LF-time"
    ]
plot_freqs = [
        "LF-timeLF-time",
        "LF-timeBeta",
        "LF-timeDelta",
        "LF-timeGamma"
    ]

plot_labels = [
        "LF-time",
        "+Beta",
        "+Delta",
        "+Gamma"
    ]

#time windows to be extracted by event here we use 1 sencond after stimulus onset and 1 second
#before button press 
time_windows=[[0,1000],[-1000,0]]
#list of participants included in the analysis
participant_list=["P01","P02","P03","P04","P05","P06"]
#path to participant data
path_to_data=str(path_data)


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
        for i_f,freq in enumerate(freq_primary):
            for i_p,par in enumerate(participant_list):
                #here the features and corresponding time-window information is loaded
                path=path_to_data+"/"+par+"/"
                data=np.load(path+"output/"+par+event+freq+str(time_windows[i_e][0])+"_until_"+str(time_windows[i_e][1])+"features.npz")
                features=data["features"]
                time_windows_temp=data["time_array"]
                for i_f2, freq2 in enumerate(freq_list):
                    data2=np.load(path+"output/"+par+event+freq2+str(time_windows[i_e][0])+"_until_"+str(time_windows[i_e][1])+"features.npz")
                    features2=data2["features"]
                    time_windows_temp2=data2["time_array"]
                    #we combine features from the primary and the other frequencies
                    features_combined=np.concat((features[:,:,:-1],features2),axis=2)
                    print(features_combined.shape)
                    features_new=copy.deepcopy(features_combined)
                    #we only want to compare informed vs. uninformed decision making so we only include
                    #first and third stimulus presentation and exclude the trials of the second stimulus presentation
                    indices_1_3=my_find(features_new[:,0,-1],'!=',2)[0]
                    mat_class=features_new[indices_1_3,:,:]
                    #remove the last column for the features where label information is saved, features without labels are saved in X
                    temp=mat_class[:,:,:-1]
                    X=temp.reshape(temp.shape[0],-1)
                    #label information is saved in y
                    y=mat_class[:,0,-1]
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
                        frequency_results_all.append(freq+freq2)
                        participant_results_all.append(par)
                        cv_number_results_all.append(k)
                        auc_values_results_all.append(auc)
                        acc_values_results_all.append(accuracy)
    #we summarize all information in data frames
    #df_prediction_results saves information for every cross-validation iteration, this is important for significance testing
    df_prediction_results=pd.DataFrame({"event":event_results_all,"frequency":frequency_results_all,"participant":participant_results_all,"cv":cv_number_results_all,"acc":acc_values_results_all,"auc":auc_values_results_all})
    #we also add a column auc_centered that saves the distance of AUC from 0.5 (chance level)
    df_prediction_results['auc_centered'] = df_prediction_results['auc'] - 0.5
    #save the dataframe
    df_prediction_results.to_csv(path_to_data+"/per_event_freq_all_time_window_prediction_results_combined_features.csv")  

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
        for i_f,freq in enumerate(np.unique(frequency_results_all)):
            #take only rows corresponding to this feature
            df_prediction_freq=df_prediction_event[df_prediction_event["frequency"]==freq]
            #define the mixed linear model
            model = smf.mixedlm("auc_centered ~ 1", df_prediction_freq, groups=df_prediction_freq["participant"])
            #fit model
            result = model.fit(method="nm")
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
    df_stats_results.to_csv(path_to_data+"/per_event_freq_all_time_window_statistic_combined_results.csv")  


   

    # Compare all combinations against only one frequency band
  

    comparison_sig = {}
    comparison_pvals = {}

    for event in event_list:

        # data for current event
        df_event = df_prediction_results[
            df_prediction_results["event"] == event
        ].copy()

        freq_order = list(np.unique(df_event["frequency"]))

       
        baseline = freq_primary[0]+freq_primary[0]

        if baseline not in freq_order:
            print(f"{baseline} not found for {event}")
            continue

        freq_order.remove(baseline)
        freq_order = [baseline] + freq_order

        df_event["frequency"] = pd.Categorical(
            df_event["frequency"],
            categories=freq_order,
            ordered=True
        )

        # mixed model
        model = smf.mixedlm(
            "auc ~ C(frequency)",
            data=df_event,
            groups=df_event["participant"]
        )

        result = model.fit(method="nm")

        # Bonferroni correction
        alpha_corr = 0.05 / (len(freq_order) - 1)

        # baseline itself
        comparison_sig[(event, baseline)] = False
        comparison_pvals[(event, baseline)] = np.nan

        # all other frequencies
        for freq in freq_order[1:]:

            term = f"C(frequency)[T.{freq}]"

            if term in result.pvalues:

                p = result.pvalues[term]

                comparison_pvals[(event, freq)] = p
                comparison_sig[(event, freq)] = (p < alpha_corr)

            else:

                comparison_pvals[(event, freq)] = np.nan
                comparison_sig[(event, freq)] = False

        print(f"\n{event}")
        print(result.summary())

    ### Plotting (event =Start Stim)
    # For event Start Stim we plot the results per participant and the distribution (mean and standard deviation across participants). Stars indicate that the AUCs for this feature and event significantly differ from 0.5
 ### Plotting (event =Start Stim)

event_of_interest = "Start Stim"

df_filtered = df_prediction_results[
    (df_prediction_results['event'] == event_of_interest) &
    (df_prediction_results['frequency'].isin(plot_freqs))
]

df_filtered_stats = df_stats_results[
    (df_stats_results['event'] == event_of_interest) &
    (df_stats_results['frequency'].isin(plot_freqs))
]

is_sig_list = []

for freq in plot_freqs:

    df_stats_temp = df_filtered_stats[
        df_filtered_stats["frequency"] == freq
    ]

    if len(df_stats_temp) == 0:
        is_sig_list.append(0)
        continue

    p_val = df_stats_temp["p_one_sided"].iloc[0]

    is_sig = int(p_val < 0.05 / len(plot_freqs))

    is_sig_list.append(is_sig)

df_means = df_filtered.groupby(
    ['frequency', 'participant'],
    as_index=False
)['auc'].mean()

df_means['frequency'] = pd.Categorical(
    df_means['frequency'],
    categories=plot_freqs,
    ordered=True
)

df_means = df_means.sort_values('frequency')

summary = (
    df_means.groupby('frequency', observed=False)['auc']
    .agg(['mean', 'std'])
    .reindex(plot_freqs)
    .reset_index()
)

participant_order = sorted(df_means['participant'].unique())

palette = dict(
    zip(
        participant_order,
        sns.color_palette('tab10', n_colors=len(participant_order))
    )
)

fig = plt.figure(figsize=(7, 6))
ax = plt.gca()

sns.set_theme(style="ticks")

x_positions = np.arange(len(plot_freqs))

for y in [0.4, 0.6, 0.7, 0.8, 0.9, 1]:
    plt.axhline(
        y=y,
        color='gray',
        linestyle='--',
        linewidth=0.8,
        zorder=1
    )

for x, (_, row) in zip(x_positions, summary.iterrows()):

    mean = row['mean']
    std = row['std']

    if pd.isna(mean):
        continue

    rect = Rectangle(
        (x - 0.25, mean - std),
        width=0.5,
        height=2 * std,
        facecolor='white',
        edgecolor='black',
        alpha=1,
        zorder=5
    )

    ax.add_patch(rect)

    plt.plot(
        [x - 0.25, x + 0.25],
        [mean, mean],
        color='black',
        linewidth=2,
        zorder=10
    )

for i, freq in enumerate(plot_freqs):

    df_freq = df_means[
        df_means['frequency'] == freq
    ]

    auc_counts = df_freq['auc'].value_counts()

    count = 0

    for _, row in df_freq.iterrows():

        auc_value = row['auc']

        if auc_counts[auc_value] > 1:

            jitter = 0.03

            if count == 0:
                pos = i - jitter
            else:
                pos = i + jitter

            count = 1

        else:

            pos = i

        plt.scatter(
            pos,
            auc_value,
            color=palette[row['participant']],
            s=60,
            edgecolor='black',
            linewidth=0.5,
            zorder=10
        )

#y_star = 0.95

#for i, sig in enumerate(is_sig_list):

    #if sig:

        #plt.scatter(
            #i,
            #y_star,
            #marker="*",
            #color="black",
            #s=70
        #)

y_plus = 0.91

for i, freq in enumerate(plot_freqs):

    if comparison_sig.get((event_of_interest, freq), False):

        plt.scatter(
            i,
            y_plus,
            marker="+",
            color="black",
            s=120,
            linewidths=2,
            zorder=20
        )

plt.axhline(
    y=0.5,
    color='gray',
    linestyle='-',
    linewidth=2,
    zorder=12
)

plt.xticks(
    x_positions,
    plot_labels,
    rotation=45,
    fontsize=20
)

plt.yticks(
    [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    ["","","0.5","","0.7","","0.9",""],
    fontsize=20
)

plt.ylabel('Mean AUC', fontsize=20)

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

for axis in ['left', 'bottom']:
    ax.spines[axis].set_linewidth(2.5)

plt.tight_layout()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
fig.savefig(str(path_data)+"/plots/prediction_frequencies_combined_"+event_of_interest+freq_primary[0]+"leave_one_repetition_out.pdf", dpi=700)
    
plt.show()


   ### Plotting (event =Press)
    # For event Start Stim we plot the results per participant and the distribution (mean and standard deviation across participants). Stars indicate that the AUCs for this feature and event significantly differ from 0.5
 ### Plotting (event =Start Stim)

event_of_interest = "Press"

df_filtered = df_prediction_results[
    (df_prediction_results['event'] == event_of_interest) &
    (df_prediction_results['frequency'].isin(plot_freqs))
]

df_filtered_stats = df_stats_results[
    (df_stats_results['event'] == event_of_interest) &
    (df_stats_results['frequency'].isin(plot_freqs))
]

is_sig_list = []

for freq in plot_freqs:

    df_stats_temp = df_filtered_stats[
        df_filtered_stats["frequency"] == freq
    ]

    if len(df_stats_temp) == 0:
        is_sig_list.append(0)
        continue

    p_val = df_stats_temp["p_one_sided"].iloc[0]

    is_sig = int(p_val < 0.05 / len(plot_freqs))

    is_sig_list.append(is_sig)

df_means = df_filtered.groupby(
    ['frequency', 'participant'],
    as_index=False
)['auc'].mean()

df_means['frequency'] = pd.Categorical(
    df_means['frequency'],
    categories=plot_freqs,
    ordered=True
)

df_means = df_means.sort_values('frequency')

summary = (
    df_means.groupby('frequency', observed=False)['auc']
    .agg(['mean', 'std'])
    .reindex(plot_freqs)
    .reset_index()
)

participant_order = sorted(df_means['participant'].unique())

palette = dict(
    zip(
        participant_order,
        sns.color_palette('tab10', n_colors=len(participant_order))
    )
)

fig = plt.figure(figsize=(7, 6))
ax = plt.gca()

sns.set_theme(style="ticks")

x_positions = np.arange(len(plot_freqs))

for y in [0.4, 0.6, 0.7, 0.8, 0.9, 1]:
    plt.axhline(
        y=y,
        color='gray',
        linestyle='--',
        linewidth=0.8,
        zorder=1
    )

for x, (_, row) in zip(x_positions, summary.iterrows()):

    mean = row['mean']
    std = row['std']

    if pd.isna(mean):
        continue

    rect = Rectangle(
        (x - 0.25, mean - std),
        width=0.5,
        height=2 * std,
        facecolor='white',
        edgecolor='black',
        alpha=1,
        zorder=5
    )

    ax.add_patch(rect)

    plt.plot(
        [x - 0.25, x + 0.25],
        [mean, mean],
        color='black',
        linewidth=2,
        zorder=10
    )

for i, freq in enumerate(plot_freqs):

    df_freq = df_means[
        df_means['frequency'] == freq
    ]

    auc_counts = df_freq['auc'].value_counts()

    count = 0

    for _, row in df_freq.iterrows():

        auc_value = row['auc']

        if auc_counts[auc_value] > 1:

            jitter = 0.03

            if count == 0:
                pos = i - jitter
            else:
                pos = i + jitter

            count = 1

        else:

            pos = i

        plt.scatter(
            pos,
            auc_value,
            color=palette[row['participant']],
            s=60,
            edgecolor='black',
            linewidth=0.5,
            zorder=10
        )

#y_star = 0.95

#for i, sig in enumerate(is_sig_list):

    #if sig:

        #plt.scatter(
            #i,
            #y_star,
            #marker="*",
            #color="black",
            #s=70
        #)

y_plus = 0.91

for i, freq in enumerate(plot_freqs):

    if comparison_sig.get((event_of_interest, freq), False):

        plt.scatter(
            i,
            y_plus,
            marker="+",
            color="black",
            s=120,
            linewidths=2,
            zorder=20
        )

plt.axhline(
    y=0.5,
    color='gray',
    linestyle='-',
    linewidth=2,
    zorder=12
)

plt.xticks(
    x_positions,
    plot_labels,
    rotation=45,
    fontsize=20
)

plt.yticks(
    [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    ["","","0.5","","0.7","","0.9",""],
    fontsize=20
)

plt.ylabel('Mean AUC', fontsize=20)

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

for axis in ['left', 'bottom']:
    ax.spines[axis].set_linewidth(2.5)

plt.tight_layout()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
fig.savefig(str(path_data)+"/plots/prediction_frequencies_combined_"+event_of_interest+freq_primary[0]+"leave_one_repetition_out.pdf", dpi=700)
    
plt.show()