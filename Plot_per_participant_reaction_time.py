
## Per participant response time analysis (Supplemantary Reaction time differences per participant and S2)
# - reaction time per participant and presenation (first, seconds, third stimulus presentation) is visualized
# - reaction time per participant and previous correct vs. incorrect trial is visualized and tested per participant using a ttest.

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

#load the behavioral data computed by Behavioral_data_analysis.py
behavioral_results = pd.read_csv(path_to_data + "/behaverioal_results.csv")
# We keep only valid response times
plot_df = behavioral_results.dropna(
    subset=['participant', 'label', 'time_first_response']
).copy()

# Convert labels to strings
plot_df['participant'] = plot_df['participant'].astype(str)
plot_df['label'] = plot_df['label'].astype(str)

# Order labels
label_order = ['1', '2', '3']

# Order participants
participant_order = plot_df['participant'].unique()

#####
#reaction time per participant and presenation
####
# Compute mean and standard deviation per participant and label
participant_summary = (
    plot_df
    .groupby(['participant', 'label'])['time_first_response']
    .agg(['mean', 'std'])
    .reset_index()
)

# We initialize the plot

sns.set_theme(style="ticks")
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()

# define the offsets per presentation label

offsets = {
    '1': -0.22,
    '2':  0.00,
    '3':  0.22
}

# Plot individual trials as scatter points
for participant_idx, participant in enumerate(participant_order):

    sub_df = plot_df[
        plot_df['participant'] == participant
    ]

    for label in label_order:

        label_df = sub_df[
            sub_df['label'] == label
        ]

        x = (
            participant_idx
            + offsets[label]
            + np.random.uniform(
                -0.04, 0.04, size=len(label_df)
            )
        )

        plt.scatter(
            x,
            label_df['time_first_response'],
            color='lightgray',
            edgecolor="darkgray",
            alpha=0.6,
            s=20,
            zorder=1
        )

# Connect the participant means across presentations

for participant_idx, participant in enumerate(participant_order):

    sub_df = (
        participant_summary[
            participant_summary['participant'] == participant
        ]
        .set_index('label')
        .reindex(label_order)
    )

    x_positions = [
        participant_idx + offsets[label]
        for label in label_order
    ]

    plt.plot(
        x_positions,
        sub_df['mean'],
        color='gray',
        alpha=0.6,
        linewidth=1,
        zorder=2
    )
#we add the mean and standard deviation per label 

palette = sns.color_palette('Set2')

for label_idx, label in enumerate(label_order):

    sub_df = participant_summary[
        participant_summary['label'] == label
    ]

    # Match participant positions
    x_positions = np.array([
        participant_order.tolist().index(participant)
        + offsets[label]
        for participant in sub_df['participant']
    ])

    ax.errorbar(
        x_positions,
        sub_df['mean'],
        yerr=sub_df['std'],
        fmt='o',

        
        markersize=10,

    
        elinewidth=2,

        capsize=0,

        # take the right color per presentation label
        color=palette[label_idx],

        markeredgecolor='black',
        markeredgewidth=0.5,

        linestyle='none',
        zorder=3
    )




# Make the plots pretty


plt.ylabel('Reaction Time (s)', fontsize=16)
plt.xlabel('Participant', fontsize=16)

plt.xticks(
    range(len(participant_order)),
    participant_order,
    fontsize=14
)

plt.yticks(
    [0,0.25,0.5,0.75,1, 1.25, 1.5, 1.75, 2, 2.25,2.5,2.75],
    ["0","","0.5","","1", "", "1.5", "", "2","","2.5",""],
    fontsize=14
)

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

plt.xlim(
    -0.5,
    len(participant_order) - 0.5
)

#plt.ylim(0.9, 2)

plt.grid(False)
sns.despine()


for axis in ['left', 'bottom']:
    ax.spines[axis].set_linewidth(2)


# Font settings 

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.tight_layout()

# Save figure
fig.savefig(
    path_to_data + "/plots/behavioral_reaction_time_by_participant_label.pdf",
    dpi=700
)

plt.show()



####
#calculating and plotting previous correct versus incorrect trials
####
behavioral_results = behavioral_results.sort_values(
    ["participant", "trial"]
).copy()

# Previous trial correctness
behavioral_results["prev_corr"] = (
    behavioral_results
    .groupby("participant")["is_correct"]
    .shift(1)
    .fillna(0)
    .astype(int)
)

plot_df = behavioral_results.copy()

# Calculate previous trial correctness within each participant
plot_df["prev_corr"] = (
    plot_df
    .groupby("participant")["is_correct"]
    .shift(1)
    .fillna(0)
    .astype(int)
)

# Keep only valid response times
plot_df = plot_df.dropna(
    subset=["participant", "prev_corr", "time_first_response"]
).copy()

#compute t-values per participant
for i_p, parti in enumerate(plot_df["participant"].unique()):
   plot_parti=plot_df[plot_df["participant"]==parti]
   prev_correct1=plot_parti[plot_df["prev_corr"]==1]
   prev_correct0=plot_parti[plot_df["prev_corr"]==0]
   print(stats.ttest_ind(prev_correct1["time_first_response"],prev_correct0["time_first_response"]))


## Plotting response time per participant and previous trial correctness 0 or 1




# Convert participant to string
plot_df["participant"] = plot_df["participant"].astype(str)

# Make sure prev_corr is integer
plot_df["prev_corr"] = plot_df["prev_corr"].astype(int)


# Previous trial correctness:
# 0 = previous trial incorrect
# 1 = previous trial correct
prev_corr_order = [0, 1]

# Order participants
participant_order = plot_df["participant"].unique()


#group by paricipant and previous correctness, compute mean and std
participant_summary = (
    plot_df
    .groupby(["participant", "prev_corr"])["time_first_response"]
    .agg(["mean", "std"])
    .reset_index()
)


#initialize figure
sns.set_theme(style="ticks")

fig = plt.figure(figsize=(6, 4))
ax = plt.gca()


#define offsets for previous correctness conditions

offsets = {
    0: -0.15,
    1:  0.15
}


#iterate over participants scatter the individual reaction times
for participant_idx, participant in enumerate(participant_order):

    sub_df = plot_df[
        plot_df["participant"] == participant
    ]

    for prev_corr in prev_corr_order:

        corr_df = sub_df[
            sub_df["prev_corr"] == prev_corr
        ]

        # Jitter individual observations
        x = (
            participant_idx
            + offsets[prev_corr]
            + np.random.uniform(
                -0.04,
                0.04,
                size=len(corr_df)
            )
        )

        plt.scatter(
            x,
            corr_df["time_first_response"],
            color="lightgray",
            edgecolor="darkgray",
            alpha=0.6,
            s=20,
            zorder=1
        )



# Connect participant means across prev_corr levels

for participant_idx, participant in enumerate(participant_order):

    sub_df = (
        participant_summary[
            participant_summary["participant"] == participant
        ]
        .set_index("prev_corr")
        .reindex(prev_corr_order)
    )

    x_positions = [
        participant_idx + offsets[prev_corr]
        for prev_corr in prev_corr_order
    ]

    plt.plot(
        x_positions,
        sub_df["mean"],
        color="gray",
        alpha=0.6,
        linewidth=1,
        zorder=2
    )


#we plot the mean and standard deviation per participant and condition as errorbars
palette = [
    "#D66A6A",  # prev_corr = 0
    "#63A875"   # prev_corr = 1
]
for corr_idx, prev_corr in enumerate(prev_corr_order):

    sub_df = participant_summary[
        participant_summary["prev_corr"] == prev_corr
    ]

    # Match participant positions
    x_positions = np.array([
        participant_order.tolist().index(participant)
        + offsets[prev_corr]
        for participant in sub_df["participant"]
    ])

    ax.errorbar(
        x_positions,
        sub_df["mean"],
        yerr=sub_df["std"],
        fmt="o",
        markersize=10,
        elinewidth=2,
        capsize=0,
        color=palette[corr_idx],
        markeredgecolor="black",
        markeredgewidth=0.5,
        linestyle="none",
        zorder=3
    )


#make the plots prettier
plt.ylabel(
    "Reaction Time (s)",
    fontsize=16
)

plt.xlabel(
    "Participant",
    fontsize=16
)

plt.xticks(
    range(len(participant_order)),
    participant_order,
    fontsize=14
)

plt.yticks(
    [
        0,
        0.25,
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        2.75
    ],
    [
        "0",
        "",
        "0.5",
        "",
        "1",
        "",
        "1.5",
        "",
        "2",
        "",
        "2.5",
        ""
    ],
    fontsize=14
)

plt.title(
    "",
    loc="left",
    fontsize=14,
    weight="bold"
)


# ---------------------------------------------------------
# Tick parameters
# ---------------------------------------------------------

plt.tick_params(
    axis="both",
    which="both",
    direction="out",
    length=2,
    width=1,
    bottom=True,
    left=True,
    top=False,
    right=False
)

#axis limits
plt.xlim(
    -0.5,
    len(participant_order) - 0.5
)


plt.grid(False)

sns.despine()

for axis in ["left", "bottom"]:
    ax.spines[axis].set_linewidth(2)


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"


plt.tight_layout()


#save figure
fig.savefig(
    path_to_data
    + "/plots/behavioral_reaction_time_by_participant_prev_corr.pdf",
    dpi=700
)

plt.show()