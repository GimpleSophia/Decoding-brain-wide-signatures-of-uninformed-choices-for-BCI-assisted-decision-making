## 3D brain plots color by ROI
# - Plot all electrodes
# - color contacts located in a specific ROI

from pathlib import Path
import numpy as np
import sys
sys.path.append("../brainplots_skripts/")
import brainplots
from collections import defaultdict
import pandas as pd
import seaborn as sns
from config import path_data


## Define some parameters:
    #################
#select all participants to be included in this analysis
participants = ["P01","P02","P03","P04","P05","P06"]

#define whether both or only one hemisphere is selected. Choose "L","R" or "both"
HEMS="both"
#this dictionary defines in which color (index in the color palette) the region of interest should be plotted. We use the color-palette Dark2.
colors=sns.color_palette("Dark2")
color_dict={"oc-temp":1,"Hippocampus":3,"white":0,"front":2,"ins":5,"ant_ins":6,"post_ins":7}

#define the region of interest to be highlighted in color, choose any from the color_dict, if choosing ant_ins or post_ins both regions of interest are plotted in different colors
ROI="ant_ins"
    #################

if __name__=='__main__':
    
    # Define paths & initialize mni surface
    file_left_hemisphere  = path_data/'MNI'/'lh_pial.mat'
    file_right_hemisphere = path_data/'MNI'/'rh_pial.mat'
    path_cortical_regions = path_data/'MNI'/'label'/'gyri'
    path_subcortical_regions = path_data/'MNI'/'subcortical'
    if HEMS=="both":
        brain = brainplots.Brain(id_= 'MNI', file_left=  file_left_hemisphere, file_right= file_right_hemisphere)
        brain.full.opacity = 0.1
    elif HEMS=="L":
        brain = brainplots.Brain(id_= 'MNI', file_left=  file_left_hemisphere)
        brain.full.opacity = 0.15 
    elif HEMS=="R":
        brain = brainplots.Brain(id_= 'MNI', file_right=  file_right_hemisphere)
        brain.full.opacity = 0.15 



        
    # Loop over participants

    contacts_list=[]
    for p_i,ppt_id in enumerate(participants):
            #load channel information of all channels
            file_name='location_struct_insula_labels_'+ppt_id+'.csv'
            path_channels = path_data/f'{ppt_id}'/"output"/file_name
            channel_struct=pd.read_csv(path_channels)

            contacts_arr = [] 
            #get the contacts structure and fill it
            contacts = brainplots.Contacts("")  
            contacts.names=np.array(channel_struct["channel_name"])
            contacts.xyz=np.array(channel_struct[["x_MNI","y_MNI","z_MNI"]])

            #initialize a color and size array
            color_arr = np.zeros((len(channel_struct), 3), dtype=float)
            size_arr = np.zeros((len(channel_struct)), dtype=float)  

            #iterate over channels
            for i, name in enumerate(contacts.names):
                #take the relevant row
                dataframe_temp=channel_struct[channel_struct["channel_name"]==name]
                
                #if neither anterior nor posterior insula are plotted
                if ROI != "ant_ins" and ROI !="post_ins":
                        #we select all channels based on the given string and the channel location and set the color accordingly
                        if ROI.lower() in dataframe_temp["location"].iloc[0].lower():
                            color_arr[i] = colors[color_dict[ROI]]
                            size_arr[i] = 5
                        else:
                            #channels not belonging to the roi are plotted in gray and smaller
                            color_arr[i] = [0.8,0.8,0.8]
                            size_arr[i] = 3
                #in case anterior and posterior insula should be plotted we look at the label is_anterior insula
                # anterior and posterior channels are plotted in different colors         
                elif dataframe_temp["is_anterior_insula"].iloc[0]==1:
                        color_arr[i] = colors[color_dict["ant_ins"]]
                        size_arr[i] = 5
                elif dataframe_temp["is_anterior_insula"].iloc[0]==0:
                        color_arr[i] = colors[color_dict["post_ins"]]
                        size_arr[i] = 5
                else:
                        #again all other channels are plotted smaller and in gray
                        color_arr[i] = [0.8,0.8,0.8]
                        size_arr[i] = 3

            #in case only one Hemisphere should be plotted we select only the relevant channels otherwise we plot all
            if HEMS !="both":
                mask = np.char.startswith(np.array(contacts.names, dtype=str), HEMS)

                select = np.where(mask)[0]                     
                        
            else:

                select=range(0,len(contacts.names))
            #we remove not selected contacts
            contacts.xyz=contacts.xyz[select]
            contacts.names=contacts.names[select]
            contacts.add_color(np.array(color_arr)[select])
            contacts.add_size(np.array(size_arr)[select])
            
            contacts_list.append(contacts)
            
    #show the plot
    scene = brainplots.plot(brain, contacts=contacts_list, show=True)

