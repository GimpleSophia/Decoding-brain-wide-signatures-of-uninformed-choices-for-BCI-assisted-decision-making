# Readme- Decoding brain-wide signatures of uninformed choices for BCI assisted decision-making

The paper corresponding to the code can be found at ....
The data corresponding to the code can be found at ...


## Packages for analysis skripts
- numpy (2.1.3)
- pandas (2.2.3)
- matplotlib (3.10.0)
- scipy (1.15.3)
- seaborn (0.13.2)
- scikit-learn (1.6.1)
- statsmodels (0.14.4)

## Packages 3D brain plotting skripts (located in forlder brainplots_skripts)
- python (3.8.18)
- numpy (1.24.4)
- mayavi (4.8.2)
- pyqt5 (5.15.11)
- vtk (9.2.2)
- scipy (1.10.1)

(if mayavi installation on Windows fails try: pip install numpy, pip install wheel==0.45.1, pip install mayavi --no-cache-dir --verbose --verbose --verbose --no-build-isolation or refer to mayavi installation instructions https://docs.enthought.com/mayavi/mayavi/installation.html)

## Getting started 
- Update the config file (config.py) set path_data to the data location on your laptop

## Skript overview and order
- helper_functions.py provides functions used in the analysis skripts, brainplots.py is used for creating 3D brainplots

### Analysis skripts
These skripts use functions defined in helper_functions.py

- Load_data_Extract_features.py --> Skript for extracting oscillatory and time domain features, run for time_windows=[[-250,1500],[-1250,250]] and 				time_windows=[[0,1000],[-1000,0]], it saves feature matrices loaded by Significance_testing... skripts

- Find_anterior_channels.py --> 	Visualizes insula regions and labels insular channels into anterior and posterior, the updated location_struct is loaded by the other skripts

- Behavioral_data_analysis.py--> Computes behavioral outcome measures (task decision accuracy and response time) and compares different label conditions, creates Fig 1 c and d

- Significance_testing_all_time_intervals_per_frequency.py (Loads files created in Load_Data_Extract_Features.py) --> File that performs feature-wise classification and significance testing using all channels and 		time intervals in time frame [[0,1000],[-1000,0]], creates Fig. 2 b and c

- Significance_testing_rois_per_frequency.py (Loads files created in Load_Data_Extract_Features.py and Find_anterior_channels.py) --> File that performs classification by time-interval ([[-250,1500],[-1250,250]]) for selected event-feature-roi 				combinations, creates Fig. 3 e and Fig. 4 b

- Significance_testing_single_channel_one_roi.py (Loads files created in Load_Data_Extract_Features.py and Find_anterior_channels.py) --> File that performs classification and significance testing for single channels and selected event-feature-roi-combinations (here done for Press-gamma-insula) for all time intervals in time frame [[0,1000],[-1000,0]] combined, creates input for Plot_insula_single_channel_t_values.py

### Plotting skripts
These skripts use classes and functions defined in brainplots.py

- Plot_electrodes_per_participant.py --> Plots all contacts colored by participant, creates Fig. 1 b

- Plot_electrodes_per_roi.py (Loads Files created in Find_anterior_channels.py)--> Plots all contacts, highlighting contacts located in one of the predefined rois, creates Fig. 3 a-d and Fig. 4 a

- Plot_insula_single_channel_t_values.py (Loads Files created in Find_anterior_channels.py and Significance_testing_single_channel_one_roi.py)-->Plots insular contacts on the insula MNI surface, highlighting anterior and posterior insular channels with 		different colors and plotting the contact size in proportion to the t-value from single channel t-testing, It also creates the distribution of all channels and channels with t>2 on an anterior-posterior scale creates all subfigures for Fig.4 c


