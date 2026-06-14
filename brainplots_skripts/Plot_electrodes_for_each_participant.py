## 3D brain plots color by participant
#  - Plot multiple participants for a group-level electrode coverage
# - Different colors correspond to different participants

from pathlib import Path
import numpy as np
import sys

sys.path.append("../brainplots_skripts/")
import brainplots
import pandas as pd
import seaborn as sns
from itertools import repeat
sys.path.insert(0, '..')
from config import path_data


## Define some parameters
    #################
#select all participants to be plotted
participants = ["P01","P02","P03","P04","P05","P06"]
#define whether both or only one hemisphere is selected. Choose "L","R" or "both"
#HEMS="both"   
HEMS="both"   
#colors are selected from the "tab10" colormap
colors=sns.color_palette("tab10")
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

    
    for p_i,ppt_id in enumerate(participants):
            contacts_list=[]
            contacts_arr = [] 
            # Load electrode coordinates and locations
            file_name='location_struct_'+ppt_id+'.csv'
            path_channels = path_data/f'{ppt_id}'/file_name
            channel_struct=pd.read_csv(path_channels)
            channel_struct["electrode"] = channel_struct["channel_name"].str[:2]
            #get the contacts structure and fill it
            contacts = brainplots.Contacts("") 
            contacts.names=np.array(channel_struct["channel_name"])
            contacts.xyz=np.array(channel_struct[["x_MNI","y_MNI","z_MNI"]]) 
            #define a color and size array for the contact plotting
            color_arr=[] 
            color_arr.extend(repeat(colors[p_i],len(channel_struct)))   
            size_arr=[]
            size_arr.extend(repeat(3,len(channel_struct)))  

            #in case only one Hemisphere should be plotted we select only the relevant channels otherwise we plot all
            if HEMS !="both":
                mask = np.char.startswith(np.array(contacts.names, dtype=str), HEMS)

                # Get the indices where mask is True
                select = np.where(mask)[0]                     
            else:
                select=range(0,len(contacts.names))
            #we remove not selected contacts
            contacts.xyz=contacts.xyz[select]
            contacts.names=contacts.names[select]
            contacts.electrodes=np.array(channel_struct["electrode"][select])
            contacts.add_color(np.array(color_arr)[select])
            contacts.add_size(np.array(size_arr)[select])
            contacts.interpolate_electrodes()
            contacts_list.append(contacts)
            #show the plot
            scene = brainplots.plot(brain, contacts=contacts_list, show=True)

