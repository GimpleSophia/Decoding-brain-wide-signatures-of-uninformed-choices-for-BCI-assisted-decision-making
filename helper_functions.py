#import some packages
import pyxdf as pyx
import copy
import numpy as np
import scipy.signal as scisig
from scipy import fftpack
import statsmodels.formula.api as smf
from scipy.stats import norm
import pandas as pd



def my_find(arr,operator, number):
    """
    This function finds the indices in an array that fulfill the condition
    Parameters:
        arr : 1 d np.ndarray
        operator : string, "<", ">", "=" or "!="
        number : int

    Returns:
        indices_out: 1d np.array of indices
        
    """
    arr=np.array(arr)
    if operator=='>':
        indices=arr>number 
      
    elif operator=='<':
        indices=arr<number
       
    elif operator=='=' or operator=='==':
        indices=(arr==number)
    
    elif operator=='!=' or operator =='~=':
        indices=(arr!=number)
  
    else:
        print("error!")
        indices=np.array([])
    
    indices_out=indices.nonzero
    indices_out=indices_out()
    return indices_out


def find_point(arr,point):
    """
    This function finds the position in a sorted array that is closest to the value of a given point
    Parameters:
        arr : 1 d np.ndarray sorted
        point : int

    Returns:
        pos_new: int value of the point
        ind: int index of the point

        
    """
    pos=np.searchsorted(arr,point)
    if pos==0:
        pos_new=arr[0]
        ind=0
    elif pos>len(arr)-1:
        pos_new=arr[-1]
        ind=-1
    elif (abs(arr[pos-1]-point)< abs(arr[pos]-point)):
        pos_new=arr[pos-1]
        ind=pos-1
    elif (abs(arr[pos-1]-point)>= abs(arr[pos]-point)):
        pos_new=arr[pos] 
        ind=pos
    return pos_new, ind

def extract_labels_from_markers(markers):
        """
        This function computes labels (informed=third stimulus presentation or uninformed= first stimulus presentation decision making) from the task markers
        Parameters:
                    markers: list of task markers, each marker text is saved in a char array

        Returns:
                    labels: list of integares 1,2,3 corresponding to each trial, 1,2 and 3 correspond to first, second and third stimulus presentation in this trial
                    labels_all: list of integers 0,1,2,3 corresponding to every marker, 0 is used to capture no label information in this marker

        """
        #initialize labels and labels_all list
        labels_all=[]
        labels=[]
        #iterate over all markers
        for h in range(len(markers)):
                    #Stimulus presentation is always represented with a string of format e.g.  ['Start Stim n.: 43 - Rep n.: 1'], the last part indicates whether it is the first
                    # second or third stimulus presentation
                    # if it is the first presentation we add a 1         
                    if 'Rep n.: 1' in markers[h][0]:
                        labels.append(1)
                        labels_all.append(1)
                    # if it is the second presentation we add a 2  
                    elif 'Rep n.: 2' in markers[h][0]:
                        labels.append(2)
                        labels_all.append(2)
                    # if it is the first presentation we add a 3  
                    elif 'Rep n.: 3' in markers[h][0]:
                        labels.append(3)
                        labels_all.append(3)
                    #if the current marker does not correspond to the stimulus presentation and does not have
                    #any information about the presentation number we add a 0 to labels_all but not to labels
                    elif markers[h][0]=="":
                        labels_all.append(-1)
                    else:
                        labels_all.append(0)
    
        return labels,labels_all
#function for rereferencing the EEG signal
def ESR_reref(eeg,channel_arr):
    """
        This function rereferences the EEG signal to the mean of the elctrode shaft
        Parameters:
                    eeg: EEG signal of shape (number recorded time points, number channels)
                    channel_arr: List of channel names coded as e.g. LX1 corresponding to L-left hemisphere, X-electrode X, 1-first contact

        Returns:
                    eeg_reref: rereferenced EEG signal
                    eeg_reref_new: rereferenced EEG signal, excluding non EEG channels e.g. EKG channels
                    channel_arr_new: List of corresponding channel names only including EEG channels
        """
    eeg_reref=copy.deepcopy(eeg)
    electrode_name=[]
    discard_arr=[]
    #iterate over all channels
    #first we want to find all unique EEG electrodes
    for x in range(len(channel_arr)):
        el_na=channel_arr[x]
        #check if the electrode name starts with an R or L, this indicates that this is an EEG channel
        if el_na[0]=="R" or el_na[0]== "L":
            #if the electrode is not yet in list electrode_name we add the electrode name (first two letters of channel name) to the list
            if not el_na[0:2] in electrode_name:
                electrode_name.append(el_na[0:2])
        #channels that are not EEG channels are listed in discard_arr
        else:
            print("Warning: There are non-sEEG channels that can not be rereferenced")
    print(discard_arr)
    #print(np.array(channel_arr)[np.array(discard_arr)])
    #for each electrode we substract the electrode mean from all channels corresponding to the electrode
    #iterate over electrodes
    for el_string in electrode_name:
        print(el_string)
        #find all corresponding channels
        ind=np.flatnonzero(np.core.defchararray.find(channel_arr,el_string)!=-1)
        #find the signal of all channels of that electrode and compute the mean over all channels per time-point
        mat=eeg_reref[:,ind]
        mean_el=np.mean(mat,1)
        #substract the mean of the electrode from the signal recorded from each contact
        eeg_reref[:,ind]=(eeg_reref[:,ind].transpose()-mean_el).transpose()
    
    return eeg_reref


hilbert3 = lambda x: scisig.hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]
def hilbert_transform_and_envelope(eeg_signal):
    """
        This function computes the envelope of the hilbert transform for each channel
        Parameters:
                    eeg_signal: EEG signal of shape (number recorded time points, number channels,number of distinct frequencies)
                    channel_arr: List of channel names coded as e.g. LX1 corresponding to L-left hemisphere, X-electrode X, 1-first contact

        Returns:
                    eeg_reref: rereferenced EEG signal
                    eeg_reref_new: rereferenced EEG signal, excluding non EEG channels e.g. EKG channels
                    channel_arr_new: List of corresponding channel names only including EEG channels
    """
    for x in range(0,eeg_signal.shape[2]):
        eeg_signal_out=copy.deepcopy(eeg_signal)
        for i in range(0,eeg_signal.shape[1]):
            signal=eeg_signal[:,i,x]
            analytic_signal = hilbert3(signal) #mit nullen auf naechste Zweierpotenz verlaengern um es zu beschleunigen
            amplitude_envelope = np.abs(analytic_signal)
            eeg_signal_out[:,i,x]=amplitude_envelope
    return eeg_signal_out


def extract_epoched_matrices_from_different_time_intervals_new(eeg_struct, eeg_ts,labels,event, time_interval_arr, markers):
    """
        This function epochs the filtered (and hilbert transformed) signal according to the given time interval in relation to the difined event and computes the mean over the time interval per channel
        Parameters:
                    eeg_struct: preprocessed eeg data of shape (number time-points, number channels,1)
                    eeg_ts: timestamps corresponding to the time points in the eeg of shape (number of time-points)
                    labels: labels capturing the stimulus presentation number (first stimulus presentation=uninformed decision-making, second stimulus presentation, third stimulus presentation=informed decision-making)
                    event: string of the relevant event
                    time_interval_arr: list with tupels of relevant time-intervals relative to the event onset
                    markers: np.array of arrays of strings, 

        Returns:
                    feature_matrix: returns a matrix with the extracted features of shape (number trials,number channels, number time-points+1)
    """
    #initialize the feature matrix
    feature_matrix=0*np.empty((len(my_find(labels, '>',0)[0]),eeg_struct.shape[1],len(time_interval_arr)+1))
    #iterate over all time-intervals
    for t_int in range(0,len(time_interval_arr)):
            count=0
            #extract the current starting and end time-interval time
            time_bef_aft_start=time_interval_arr[t_int][0]  
            time_bef_aft_end=time_interval_arr[t_int][1]   
            #find all indices corresponding to the event and to the end of the trial
            indices_event = np.where(np.char.find(markers, event) >= 0)[0]
            indices_end_trial = np.where(np.char.find(markers, "Sum Trail") >= 0)[0]
            #for every label we want to find the closest occurence of the event that happens before the trial ends
            #we iterate over the labels corresponding to trials
            for lab_ind in np.nditer(my_find(labels, '>',0)):
                    #we extract and add the current label (1,2,3- number of stimulus presentation to the feature matrix)
                    lab=labels[lab_ind]
                    feature_matrix[count,:,-1]=lab
                    #we find all event and end trial indeces that happen after the event
                    larger_event_index = indices_event[indices_event >= lab_ind]
                    larger_end_trial_index = indices_end_trial[indices_end_trial >= lab_ind]
                    #in case there are events happening after the label
                    if len(larger_event_index) > 0:
                        # we find the index of the closest event after the label
                        closest_event_idx = larger_event_index[np.argmin(larger_event_index - lab_ind)]
                        
                        # in case there is a trial ending after the label (sanity check) we find the index of the closest end of trial after the label
                        if len(larger_end_trial_index) > 0:
                            # we find the index of the closest trial ending after the label
                            closest_end_trial_idx = larger_end_trial_index[np.argmin(larger_end_trial_index - lab_ind)]

                            # we check that the closest event occured in the same trial as the label (before the corresponding end trial event)
                            if closest_end_trial_idx < closest_event_idx:
                                #if this is not the case we print a warning
                                print("Warning: No response given (end of trial before next event)")
                            else:
                                #Otherwise we find the indices of the current start and end time-interval points in correspondence to the current event index
                                [eeg_start,eeg_start_ind]=find_point(eeg_ts,eeg_ts[closest_event_idx]+(time_bef_aft_start/1000))
                                [eeg_end,eeg_end_ind]=find_point(eeg_ts,eeg_ts[closest_event_idx]+(time_bef_aft_end/1000))
                                
                                #we extract the corresponding time-points of the eeg structure
                                signal=eeg_struct[eeg_start_ind:eeg_end_ind,:,0]
                                #we compute the mean of the time interval for every channel
                                sig_new=signal.mean(axis=0)
                                #we insert the computed features at the right position in the feature-matrix
                                feature_matrix[count,:,t_int]=sig_new

                        else:
                            #we give a warning in case there is no corresponding end of trial event
                            print("Warning: No end-of-trial index found.")
                    else:
                        #we give a warning in case there is no event marker after this label
                        print("Warning: The event does not occur after this label")
                    count=count+1
    return feature_matrix



def method_to_extract_new_sampling_updated_p_press_flex(eeg_struct,eeg_sr,eeg_ts,labels,event,time_interval_arr,sr_resample,markers):
    """
        This function epochs the filtered signal according to the given time interval in relation to the difined event and extracts a down-sampled signal of the time interval per channel
        Parameters:
                    eeg_struct: preprocessed eeg data of shape (number time-points, number channels,1)
                    eeg_sr: int sampling rate of the eeg data
                    eeg_ts: timestamps corresponding to the time points in the eeg of shape (number of time-points)
                    labels: labels capturing the stimulus presentation number (first stimulus presentation=uninformed decision-making, second stimulus presentation, third stimulus presentation=informed decision-making)
                    event: string of the relevant event
                    time_interval_arr: list with tupels of relevant time-intervals relative to the event onset
                    sr_resample: int sampling rate to be resampled to
                    markers: np.array of arrays of strings, 

        Returns:
                    feature_matrix: returns a matrix with the extracted features of shape (number trials,number channels, number time-intervals *number resampled time-point+1)
    """
    
    #initialize the new feature matrix
    feature_matrix=0*np.empty((len(my_find(labels, '>',0)[0]),eeg_struct.shape[1],(int(sr_resample/2)*len(time_interval_arr))+1))
    
    new_sr=sr_resample
    start_tint=0
    #iterate over all time-windows
    for t_int in range(0,len(time_interval_arr)):
        count=0
                    #extract the current starting and end time-interval time
        time_bef_aft_start=time_interval_arr[t_int][0]
        time_bef_aft_end=time_interval_arr[t_int][1]
        #compute the starting index and the end index (new start index) based on the new sampling rate
        start_tint_new=start_tint+int(sr_resample/2)
        #find all indices corresponding to the event and to the end of the trial
        indices_event = np.where(np.char.find(markers, event) >= 0)[0]
        indices_end_trial = np.where(np.char.find(markers, "Sum Trail") >= 0)[0]
        #for every label we want to find the closest occurence of the event that happens before the trial ends
        #we iterate over the labels corresponding to trials
        for lab_ind in np.nditer(my_find(labels, '>',0)):
             #we extract and add the current label (1,2,3- number of stimulus presentation to the feature matrix)
            lab=labels[lab_ind]
            feature_matrix[count,:,-1]=lab
            #we find all event and end trial indeces that happen after the event
            larger_event_index = indices_event[indices_event >= lab_ind]
            larger_end_trial_index = indices_end_trial[indices_end_trial >= lab_ind]
            #in case there are events happening after the label
            if len(larger_event_index) > 0:
                        # we find the index of the closest event after the label
                        closest_event_idx = larger_event_index[np.argmin(larger_event_index - lab_ind)]
                        # in case there is a trial ending after the label (sanity check) we find the index of the closest end of trial after the label
                        if len(larger_end_trial_index) > 0:
                            # we find the index of the closest trial ending after the label
                            closest_end_trial_idx = larger_end_trial_index[np.argmin(larger_end_trial_index - lab_ind)]

                            # we check that the closest event occured in the same trial as the label (before the corresponding end trial event)
                            if closest_end_trial_idx < closest_event_idx:
                                #if this is not the case we print a warning
                                print("Warning: No response given (end of trial before next event)")
                            else:
                                #Otherwise we find the indices of the current start and end time-interval points in correspondence to the current event index, we add 500ms before and after the time-interval time point to extract
                                #a longer signal for downsampling
                                [eeg_start,eeg_start_ind]=find_point(eeg_ts,eeg_ts[closest_event_idx]+((time_bef_aft_start-500)/1000))
                                [eeg_end,eeg_end_ind]=find_point(eeg_ts,eeg_ts[closest_event_idx]+((time_bef_aft_end+500)/1000))
                                #we extract the corresponding time-points of the eeg time array
                                time_p=eeg_ts[eeg_start_ind:eeg_end_ind]
                                #we extract the corresponding time-points of the eeg structure
                                signal=eeg_struct[eeg_start_ind:eeg_end_ind,:,0]  
                                #we downsample the eeg signal and also compute the corresponding time-points
                                q=int(eeg_sr/new_sr)    
                                signal_ds=scisig.decimate(signal,q,axis=0)
                                time_ds=time_p[::q]
                                #in the downsampled time-signal we find the time-points closest to the time-interval start and end time
                                [eeg_start_ds,eeg_start_ds_ind]=find_point(time_ds,eeg_ts[closest_event_idx]+(time_bef_aft_start/1000))
                                [eeg_end_ds,eeg_end_ds_ind]=find_point(time_ds,eeg_ts[closest_event_idx]+(time_bef_aft_end/1000))
                                #we extract the relevant points of the downsampled signal
                                signal_new=signal_ds[eeg_start_ds_ind:eeg_end_ds_ind,:]
                                #we also extract the relevant points of the downsampled timestamps
                                time_new=time_ds[eeg_start_ds_ind: eeg_end_ds_ind]
                                #in case the extracted signal is longer than expected we only take the expected amount e.g. 4 for sr_downsample=8 and time-interval of length 500ms
                                signal_t=signal_new[0:int(sr_resample/2),:]
                                #we insert the singal points at the right position in the feature matrix
                                feature_matrix[count,:,start_tint:start_tint_new]=signal_t.transpose()
                                count=count+1
                                
                        else:
                            #we give a warning in case there is no corresponding end of trial event
                            print("Warning: No end-of-trial index found.")
            else:
                        #we give a warning in case there is no event marker after this label
                        print("Warning: The event does not occur after this label")

        start_tint=start_tint_new

    return feature_matrix


def fit_lmm_stats_one_sided(group, baseline=0.5):
    """
    Linear mixed model test whether the mean AUC is significantly larger than the baseline.
    H0: mean == baseline
    H1: mean > baseline
            Parameters:
                    group: dataframe grouped by frequency, roi and time to be tested
                    baseline: chance level to be tested against (0.5)

        Returns:
                    pd_series with statistical values: coefficient (b), standard error (se), z-value (z), one sided p value (p_one_sided), 95% confidence level (ci_low, ci_high) and mean auc(mean_auc)
    """
    group = group.copy()
    group['auc_centered'] = group['auc'] - baseline

    try:
        #define the model
        model = smf.mixedlm("auc_centered ~ 1", group, groups=group["participant"])
        #model = smf.mixedlm("auc_centered ~ 1", data=group)
        #fit the model
        result = model.fit(method="nm")
        #This can throw a warning but we want to double check that the model actually converged
        if result.converged:
            print("Model converged successfully")
        else:
            print("Model did NOT converge")
        #compute the corresponding parameters
        b = result.params['Intercept']
        se = result.bse['Intercept']
        z = result.tvalues['Intercept']

        # One-sided p-value (testing b > 0)
        p_one_sided = 1 - norm.cdf(z)

        ci_low, ci_high = result.conf_int().loc['Intercept']
        mean_auc = group['auc'].mean()
        #return all values in a dataseries
        return pd.Series({
            'b': b,
            'se': se,
            'z': z,
            'p_one_sided': p_one_sided,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'mean_auc': mean_auc
        })

    except Exception:
        #In case the model fails we print an exception
        print("Exception: NAN values returned")
        return pd.Series({
            'b': np.nan, 'se': np.nan, 'z': np.nan, 'p_one_sided': np.nan,
            'ci_low': np.nan, 'ci_high': np.nan,
            'mean_auc': group['auc'].mean()
        })
