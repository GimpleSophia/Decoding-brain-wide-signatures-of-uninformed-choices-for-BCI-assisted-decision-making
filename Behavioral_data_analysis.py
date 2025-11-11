## Behavioral data analysis (accuracy and response time)
# - computes Accuracy and response times for all the participants
# - tests for significant differences between first(uninfomred), second and third(informed) stimulus presentation

#import packages and help functions form skript helper_functions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from helper_functions import *
from config import path_data


##  Define some settings
#participants to be included
participant_list=["P01","P02","P03","P04","P05","P06"]
#data files are saved in this path imported from the config file
path_to_data=str(path_data)


if __name__=='__main__':
    
    #We want to save all behavioral data in one dataframe for this we initialize some arrays to intermediately save the data
    # - list of labels (1st stimulus presentation (1)=uninformed decision-making, 2nd stimulus presentation (2), 3rd stimulus presentation (3)= informed decision-making)
    labels=[]
    # - list of current participant
    participant_list_results=[]
    # - list whether participant gave the correct answer
    correct_answer=[]
    # - list of trial numbers
    trial_number=[]
    # - time until first press
    time_arr_first=[]

    #number of trials with an too early press
    wrong_press_trials=0

    #marker that is set to 1 when a new trial is started
    trial_is_started=0
    #we iterate over particiapnts
    for i_p,par in enumerate(participant_list):

        #load the data from the participant's data folder
        path=path_to_data+par
        #we load the markers
        markers=np.load(path_to_data+"/P0"+str(i_p+1)+"/markers_P0"+str(i_p+1)+".npy")
        #we load the eeg timestamps which are the same as the marker timestamps
        marker_ts=np.load(path_to_data+"/P0"+str(i_p+1)+"/sEEG_timestamps_P0"+str(i_p+1)+".npy")
        #we iterate over markers
        for h in range(len(markers)):
                        #at the start of the trial
                        if "Start Trial" in markers[h][0]:
                                #we append information about the current participant and trial number
                                participant_list_results.append(par)
                                trial_number.append( int(''.join(filter(str.isdigit,markers[h][0]))))
                                #we set flag to 1 to mark that a new trial has started
                                trial_is_started=1
                        #When the stimulus is presented we can read information about the number/repetition of stimulus presentation
                        if 'Rep n.: 1' in markers[h][0]:
                            #we remember the start time
                            start_time=marker_ts[h]
                            #we set a flag that a stimulus was presented, the next button press will be the first
                            isfirst=1
                            #we append the presentation number to array labels
                            labels.append(1)
                        elif 'Rep n.: 2' in markers[h][0]:
                            #we remember the start time
                            start_time=marker_ts[h]
                            #we set a flag that a stimulus was presented, the next button press will be the first
                            isfirst=1
                            #we append the presentation number to array labels
                            labels.append(2)
                        elif 'Rep n.: 3' in markers[h][0]:
                            #we set a flag that a stimulus was presented, the next button press will be the first
                            isfirst=1
                            #we remember the start time
                            start_time=marker_ts[h]
                            #we append the presentation number to array labels
                            labels.append(3)
                        #if a button was pressed and we know the start time of the stimulus presentation
                        if "Press" in markers[h][0] and start_time!=np.nan:
                            #if the timing is not incorrect (faster than 1 second after stimulus presentation)
                            if "wrong time" not in markers[h][0]:
                                    trial_is_started=0
                                    #if this is the first press we append the reaction time
                                    if isfirst:
                                        time_arr_first.append(marker_ts[h]-start_time)
                                    #the next press in the same trial will not be the first press
                                    isfirst=0
                                    #we set the start_time to np.nan and wait for the next trial
                                    start_time=np.nan

                            else:
                                #if the timing is wrong we still see if this is the first press
                                if isfirst==1:
                                    #if yes we again append the reaction time
                                    time_arr_first.append(marker_ts[h]-start_time)
                                    #the next press in the same trial will not be the first press
                                    isfirst=0
                                    #we remember that this is a trial in which a too early press occured
                                    wrong_press_trials=wrong_press_trials+1
                        #at the end of the trial we find the trial summary
                        if "Sum Trail" in markers[h][0]:
                                #we give a warning if there was no correctly timed response
                                if trial_is_started==1:
                                        print("Warning:no correct response, this trial will be excluded from the behavioral accuracy analysis")

                                #if the button press (left or right) was correct we save a 1 in array correct_answer
                                if "Correct" in markers[h][0]:
                                            correct_answer.append(1)
                                #if it was incorrect we save a 0
                                elif "Incorrect" in markers[h][0]:
                                                correct_answer.append(0)
                                #if no answer we save a nan in case no answer was given
                                else:
                                    correct_answer.append(np.nan)   
                                
    #we summarize all behavioral results in a dataframe                
    behavioral_results=pd.DataFrame({"participant":participant_list_results,"trial":trial_number,"label":labels,
    "is_correct":correct_answer,"time_first_response":time_arr_first})
    # Drop rows with NaNs in critical columns
    behavioral_results = behavioral_results.dropna(subset=['is_correct', 'time_first_response', 'label'])

    ## Plotting accuracy
    # We want to plot the accuracy over the 3 presentations both as a mean over participants but also for individual participants
    
    #we group by participants and labels
    accuracy_df = (
        behavioral_results.groupby(['participant', 'label'])['is_correct']
        .mean()
        .reset_index(name='accuracy')
    )

    # We convert labels to categorical
    accuracy_df['label'] = accuracy_df['label'].astype(str)

    #we initialize the plot
    sns.set_theme(style="ticks")
    fig=plt.figure(figsize=(5, 4))
    ax = plt.gca()
    # We draw individual participant lines for accuracy over stimulus presentations
    for pid, sub_df in accuracy_df.groupby('participant'):
        sub_df = sub_df.dropna(subset=['accuracy'])
        plt.plot(sub_df['label'], sub_df['accuracy'], color='gray', alpha=0.6, linewidth=1)

    #We plot errorbars of mean accuracy and standarddeviation over participants by label
    sns.pointplot(
        data=accuracy_df,
        x='label', y='accuracy',
        hue='label',
        errorbar='sd',
        linestyles=None,
        markers='o',
        capsize=0,
        markersize=15,
        palette='Set2',
        err_kws={'linewidth': 2}
    )

    #set plotting parameters and aesthethics
    plt.axhline(0.5, linestyle='--', color='k', alpha=0.6)
    plt.ylabel('Mean Accuracy', fontsize=16)
    plt.xlabel('Presentation',fontsize=16)
    plt.xticks([0,1,2],["1","2","3"],fontsize=14)
    plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],["","","0.5","","0.7","","0.9",""],fontsize=14)
    plt.title('', loc='left', fontsize=14, weight='bold')
    plt.tick_params(
        axis='both',        # apply to both axes
        which='both',       # major and minor ticks
        direction='out',    # ticks point outwards
        length=2,           # tick length
        width=1,            # line width
        bottom=True,
        left=True,
        top=False,
        right=False
    )
    plt.xlim(-0.1,2.1)
    plt.grid(False)
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(2)
    sns.despine()
    # Ensure TrueType font embedding
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42  
    plt.rcParams['font.family'] = 'DejaVu Sans'  # or another system font
    fig.savefig(path_to_data +"/plots/behavioral_accuracy_plot.pdf", dpi=700)

    #We make a summary computing means, standard deviation and counts
    summary = (
        accuracy_df.groupby('label')['accuracy']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    print(summary)



    # We use pivot to get paired data per participant
    pivot_df = accuracy_df.pivot(index='participant', columns='label', values='accuracy')

    # We run paired t-tests between the conditions and print the results
    t_ab, p_ab = stats.ttest_rel(pivot_df['1'], pivot_df['2'])
    t_ac, p_ac = stats.ttest_rel(pivot_df['1'], pivot_df['3'])
    t_bc, p_bc = stats.ttest_rel(pivot_df['2'], pivot_df['3'])

    print(f"1 vs 2: t={t_ab:.3f}, p={p_ab:.4f}")
    print(f"1 vs 3: t={t_ac:.3f}, p={p_ac:.4f}")
    print(f"2 vs 3: t={t_bc:.3f}, p={p_bc:.4f}")


    ## Plotting response time
    # We also want to plot the response time over the 3 presentations both as a mean over participants but also for individual participants

    #we group by participants and labels
    first_time_df = behavioral_results.groupby(['participant', 'label'])['time_first_response'].mean().reset_index()
    # We convert labels to strings
    first_time_df['label'] = first_time_df['label'].astype(str)  

    #we initialize the plot
    sns.set_theme(style="ticks")
    fig=plt.figure(figsize=(5, 4))
    ax = plt.gca()
    # We draw individual participant lines for response time over stimulus presentations
    for pid, sub_df in first_time_df.groupby('participant'):
        sub_df = sub_df.dropna(subset=['time_first_response'])
        plt.plot(sub_df['label'], sub_df['time_first_response'], color='gray', alpha=0.6, linewidth=1)

    #We plot errorbars of mean and standard deviation response time over participants by label
    sns.pointplot(
        data=first_time_df,
        x='label', y='time_first_response',
        hue='label',
        errorbar='sd',
        linestyles=None,
        markers='o',
        capsize=0,          # keep capsize 0
        markersize=15,
        palette='Set2',
        err_kws={'linewidth': 2}
    )

    #set plotting parameters and aesthethics
    plt.ylabel('Mean Reaction Time (s)', fontsize=16)
    plt.xlabel('Presentation',fontsize=16)
    plt.xticks([0,1,2],["1","2","3"],fontsize=14)
    plt.title('', loc='left', fontsize=14, weight='bold')
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

    plt.xlim(-0.1, len(first_time_df['label'].unique()) - 0.9)
    plt.xticks([0,1,2],["1","2","3"],fontsize=14)
    plt.yticks([1,1.25,1.5,1.75,2],["1","","1.5","","2"],fontsize=14)
    plt.ylim(0.9,2)
    plt.grid(False)
    sns.despine()
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(2)
    # We enbed text in the figure
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42  
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig.savefig(path_to_data+"/plots/behavioral_reaction_time_plot.pdf", dpi=700)
    plt.show()


    #We make a summary computing means, standard deviation and median
    summary_stats = behavioral_results.groupby('label')['time_first_response'].agg(['mean', 'std','median']).reset_index()
    print(summary_stats)


    #We use a linear mixed effects model to test for differences between labels while using particiopant as a grouping variable
    lme = smf.mixedlm("time_first_response ~ label", data=behavioral_results, groups=behavioral_results["participant"])
    #we fit the model
    lme_result = lme.fit()
    # Coefficients table
    coefs = lme_result.summary().tables[1]
    print(coefs)

    # We use tukey posthoc tests to test for differneces between label pairs
    tukey = pairwise_tukeyhsd(
        endog=behavioral_results['time_first_response'],  # dependent variable
        groups=behavioral_results['label'],              # factor
        alpha=0.05
    )
    #we print the summary
    print(tukey.summary())


    # We summarize differences between label 1 and 3 over all participants
    #for this we keep only label 1 and 3
    subset_df = behavioral_results[behavioral_results['label'].isin([1, 3])]

    # We compute the mean per participant and label
    participant_means = subset_df.groupby(['participant', 'label'])['time_first_response'].mean().reset_index()

    # We pivot so labels are columns
    pivot_df = participant_means.pivot(index='participant', columns='label', values='time_first_response')
    pivot_df = pivot_df.rename(columns={1: 'label_1', 3: 'label_3'})
    pivot_df["diff_1_3"] = pivot_df["label_3"] - pivot_df["label_1"]
    # We compute and print a table with per-participant means and differences
    full_table = pivot_df.reset_index()
    print(full_table)