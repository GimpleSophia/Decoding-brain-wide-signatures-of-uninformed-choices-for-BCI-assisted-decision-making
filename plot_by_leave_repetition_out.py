##We plot the prediction results of all three conditions 1st vs. 2nd, 2nd vs. 3rd and 1st vs 3rd stimulus presentation

##Import 
#import some packages and functions from helper_functions.py


import pandas as pd
from config import path_data
import numpy as np
from scipy.stats import sem
path_data=str(path_data)
import matplotlib.pyplot as plt
import seaborn as sns

#we load the three result dataframes
#df1 = pd.read_csv(path_data+"/per_event_freq_all_time_window_prediction_results_one_block_out1.csv") 
#df2 = pd.read_csv(path_data+"/per_event_freq_all_time_window_prediction_results_one_block_out2.csv") 
#df3 = pd.read_csv(path_data+"/per_event_freq_all_time_window_prediction_results_one_block_out3.csv") 
df1=pd.read_csv(path_data+"/per_event_freq_all_time_window_statistic_results_LOBO_1.csv")  
df2=pd.read_csv(path_data+"/per_event_freq_all_time_window_statistic_results_LOBO_2.csv")  
df3=pd.read_csv(path_data+"/per_event_freq_all_time_window_statistic_results_LOBO_3.csv")  
list_event=[]
list_freq=[]
list_number=[]
list_auc_mean=[]
list_auc_std=[]
list_sig=[]
freq_list=["Delta","Theta","Alpha","Beta","Gamma","LF-time"]
event_list=["Start Stim","Press"]
#event_list=["Start Stim"]
freq_list=["Delta","Theta","Alpha","Beta","Gamma","LF-time"]
#we iterate over events and frequencies
for i_e, event in enumerate(event_list):
    df1_event=df1[df1["event"]==event]
    df2_event=df2[df2["event"]==event]
    df3_event=df3[df3["event"]==event]
    for i_f,freq in enumerate(freq_list):
        #extract the relevant part of the dataframe
        df1_freq=df1_event[df1_event["frequency"]==freq]
        df2_freq=df2_event[df2_event["frequency"]==freq]
        df3_freq=df3_event[df3_event["frequency"]==freq]
        #we read mean, standard deviation and pvalues, and correct the pvalues using bonferroni correction
        mean_auc_1=df1_freq["auc_mean"].values[0]
        std_auc_1=df1_freq["auc_std:"].values[0]
        sig_auc1=df1_freq["p_one_sided"].values[0]<(0.05/len(df1_event))
        print(len(df1_event),len(freq_list))
        mean_auc_2=df2_freq["auc_mean"].values[0]
        std_auc_2=df2_freq["auc_std:"].values[0]
        sig_auc2=df2_freq["p_one_sided"].values[0]<(0.05/len(df2_event))
        
        mean_auc_3=df3_freq["auc_mean"].values[0]
        std_auc_3=df3_freq["auc_std:"].values[0]
        sig_auc3=df3_freq["p_one_sided"].values[0]<(0.05/len(df3_event))
        # we append the relevant values
        list_event.append(event)
        list_freq.append(freq)
        list_number.append(1)
        list_sig.append(sig_auc1)

        list_auc_mean.append(mean_auc_1)
        list_auc_std.append(std_auc_1)
        list_event.append(event)
        list_freq.append(freq)
        list_number.append(2)
        list_sig.append(sig_auc2)
        list_auc_mean.append(mean_auc_2)
        list_auc_std.append(std_auc_2)
        list_event.append(event)
        list_freq.append(freq)
        list_number.append(3)
        list_auc_mean.append(mean_auc_3)
        list_auc_std.append(std_auc_3)
        list_sig.append(sig_auc3)
    print(df3_event)

#we combine the lists of relevant values to a dataframe for plotting
df_plot = pd.DataFrame({
    "event": list_event,
    "frequency": list_freq,
    "number": list_number,
    "auc_mean": list_auc_mean,
    "auc_std": list_auc_std,
    "significant": list_sig
})


# we define some parametersfor plotting
sns.set_theme(style="ticks")
#colors
set2 = sns.color_palette("Set2").as_hex()

block_colors = {
    1: set2[0],  
    2: set2[1],  
    3: set2[2]  
}
#font sizes
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

#for each event
for event in event_list:

    df_event = df_plot[df_plot["event"] == event]

    fig, ax = plt.subplots(figsize=(7, 6))

    x = np.arange(len(freq_list))
    width = 0.25

    # grid lines

    for y in [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ax.axhline(y, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    #chance level at 50%
    ax.axhline(0.5, color="black", linestyle="-", linewidth=2, zorder=2)


    # we make bar plots and indicate significant decoding
 
    for i, number in enumerate([1, 3, 2]):

        df_num = df_event[df_event["number"] == number]

        xpos = x + (i - 1) * width

        for j, freq in enumerate(freq_list):

            row = df_num[df_num["frequency"] == freq].iloc[0]

            mean = float(row["auc_mean"])
            err = float(row["auc_std"])
            sig = bool(row["significant"])

            # bar plots
            ax.bar(
            xpos[j],
            mean,
            width,
            yerr=err,
            capsize=4,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.95,
            color=block_colors[number],   # ⭐ KEY ADDITION
            zorder=5
        )

            # star above significant bar
            if sig:
                ax.text(
                    xpos[j],
                    mean + err + 0.02,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    fontweight="bold",
                    zorder=10
                )

 #make the figures pretty
    ax.set_xticks(x,x, fontsize=20)
    ax.set_xticklabels(freq_list, rotation=45, fontsize=20)

    ax.set_ylabel("Mean AUC", fontsize=20)
    plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[0.3,"",0.5,"","0.7","",0.9,""], fontsize=19)
    ax.set_ylim(0.3, 1.0)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=4,
        width=1.5
    )

    sns.despine(ax=ax, top=True, right=True)

    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)


    plt.tight_layout()
    # save the figure
    fig.savefig(path_data+"/plots/prediction_per_frequency_all_times_"+event+"leave_one_repetition_out.pdf", dpi=700)
    plt.show()
