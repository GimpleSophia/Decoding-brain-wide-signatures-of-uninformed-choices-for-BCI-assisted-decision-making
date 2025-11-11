##Data loading and feature extractsion
# - loads the data from all 6 participants performing the Decision-making task.
# - Loaded data is preprocessed and features are extracted. 
# - The data is saved in .npz files to be loaded later for further analysis. 
# - Features are extracted aligned to Stimulus onset or Button Press in the task.

##
#import some packages and functions from skript helper_functions.py
##
import scipy.signal as scisig
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *
import os
from config import path_data

##Settings

#events in the task ("Start Stim" is the start of stimulus presentation,
#"Press" is the time point when the button is pressed)
event_list=["Start Stim","Press"]

#list of the features to extract oscillatory features
#(delta (1-3Hz,hilbert transform), theta (4-7Hz,hilbert transform), alpha(8-12Hz,hilbert transform), beta(13-30Hz,hilbert transform), gamma (30-100Hz, hilbert transform)) and low-frequency filtered (1-40Hz, no hilbert transform)
#time domain signals
freq_list=["Delta","Theta","Alpha","Beta","Gamma","LF-time"]


#for every frequency band we save the correct frequency range and information
#if this frequency band should be hilbert transformed (1 or 0). We use hilbert transform
#for oscillatory features but not for the time-domain features
freq_dict={"Delta":[(1,3),1],"Theta":[(4,7),1],"Alpha":[(8,12),1],"Beta":[(13,30),1],"Gamma":[(30,100),1],"LF-time":[(1,40),0]}
#time windows to be extracted by event, we use timewindows from 0 to 1000 and -1000 to 0 for classification over multiple time windows combined
#to better illustrate the time course of prediction AUC we extract a bigger time window form -250 to 1500 and from -1250 to 250
#time_windows=[[-250,1500],[-1250,250]]
time_windows=[[0,1000],[-1000,0]]
#sampling rate of the eeg data
eeg_sr=1024
#length of the time windows
window_length=500
#offset between timewindows
window_offset=50
#sampling rate to which the time signal is resampled when looking at LF-time. We downsample instead of averaging here to keep the
#trends of the temporal dynamics in each time window
downsample=8

#participants to be included
participant_list=["P01","P02","P03","P04","P05","P06"]
#data files are saved in this path
path_to_data=str(path_data)

##Feature extraction per participant event and frequency

if __name__=='__main__':
    
    for i_p,par in enumerate(participant_list):
        #load the data from the participant folder
        path=path_to_data+"/"+par+"/"
        eeg=np.load(path_to_data+"/P0"+str(i_p+1)+"/sEEG_data_P0"+str(i_p+1)+".npy")
        eeg_ts=np.load(path_to_data+"/P0"+str(i_p+1)+"/sEEG_timestamps_P0"+str(i_p+1)+".npy")
        channel_arr=np.load(path_to_data+"/P0"+str(i_p+1)+"/channels_P0"+str(i_p+1)+".npy")
        markers=np.load(path_to_data+"/P0"+str(i_p+1)+"/markers_P0"+str(i_p+1)+".npy")
        nyquist=eeg_sr/2
        #extract labels from the markers
        labels,labels_all=extract_labels_from_markers(markers)
        #rereference the EEG signal using ESR method (referencing to the mean of each electrode)
        eeg_reref=ESR_reref(eeg,channel_arr)
        
        #iterate over all frequencies
        #iterate over events
        for i_e, event in enumerate(event_list):
            #calculate the exact time windows and append them to time_interval_arr
            time_interval_arr=[]
            #we select the predifined time window for the specific event
            time_window_here=time_windows[i_e]
            #we start at the time specified by the first value
            start_val=time_window_here[0]
            #start point for the first time interval
            start_val_curr=start_val
            #the end value us the second specified value
            end_val=time_window_here[1]
            #end point for the first time interval
            end_val_curr=start_val_curr+window_length
            #we calculate and append the time windows until the end value of the time
            #window exceeds the predifined general end value
            while end_val_curr<=end_val:
                #we append the current time window
                time_interval_arr.append((start_val_curr,end_val_curr))
                #we compute the next start point by adding the offset
                start_val_curr=start_val_curr+window_offset
                #we compute the next end point by adding the window length to the next start point
                end_val_curr=start_val_curr+window_length
            
            #iterate over all oscillatory and time-domain features
            for i_f,freq in enumerate(freq_list):
                        #look up the right frequency band and if the frequency band should
                        #be hilbert transformed
                        freq_information=freq_dict[freq]
                        freq_tupel=[freq_information[0]]
                        freq_is_hilbert=freq_information[1]
                        
                        #prepare a new matrix the filtered signal will be saved in
                        eeg_freq_new=copy.deepcopy(eeg_reref)       
                        eeg_freq_new = eeg_freq_new.astype('float64') 
                        eeg_freq = np.zeros((eeg_freq_new.shape[0],eeg_freq_new.shape[1],len(freq_tupel)))
                        eeg_freq=eeg_freq.astype('float64')
                        
                        #filter every channel seperately using a 3rd order butterworth filter
                        for i in range(0,len(freq_tupel)):
                            b, a = scisig.butter(3, [freq_tupel[i][0]/nyquist,freq_tupel[i][1]/nyquist], btype='bp')
                            for chan in range(0,len(channel_arr)):
                                                    dat=eeg_freq_new[:,chan]
                                                    dat_new=scisig.detrend(dat,axis=0)
                                                    eeg_freq[:,chan,i]=scisig.filtfilt(b,a,dat_new)
                        #here we differentiate between features that should be hilbert transformed and not                            
                        if freq_is_hilbert:       
                        #estimate the envelope  by extracting the magnitude of the analytic signal obtained through the Hilbert transform
                            eeg_trans_freq=hilbert_transform_and_envelope(eeg_freq)
                        #extract the mean power per time_window 
                            features_temp=extract_epoched_matrices_from_different_time_intervals_new(eeg_trans_freq,eeg_ts,labels_all ,event,time_interval_arr,markers=markers)

                        else:
                        #extract downsampled time domain features, no hilbert transform is performed
                            features_temp=method_to_extract_new_sampling_updated_p_press_flex(eeg_freq,eeg_sr,eeg_ts,labels_all,event,time_interval_arr,downsample,markers=markers)
                        #we save the extracted feature for each event, feature and participant together with information about the extracted time window
                        if not os.path.exists(path+"output/"):
                            os.makedirs(path+"output/")
                        np.savez(path+"output/"+par+event+freq+str(start_val)+"_until_"+str(end_val)+"features.npz",features=features_temp,time_array=time_interval_arr)
    


                    
                    




