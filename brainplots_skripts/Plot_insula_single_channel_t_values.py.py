## Insula single channel plotting
# - Plot anterior and posterior insula channels on the outline of the right insula
# - contact size is scaled by t-value
# - We also plot the distribution of amount of channels and amount of channels with t-values above 2 over the anterior-posterior axis of the insula

from pathlib import Path
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
sys.path.insert(0, './brainplots_skripts')
import brainplots
from config import path_data

#define the participants to be included
participants = ["P01","P02","P03","P04","P05","P06"]

if __name__=='__main__':

    # Define paths & initialize mni surface
    file_left_hemisphere  = path_data/"MNI"/'lh_pial_trivert.mat'
    file_right_hemisphere = path_data/"MNI"/'rh_pial_trivert.mat'
    path_cortical_regions = path_data/"MNI"/'label'/'gyri'
    path_subcortical_regions = path_data/"MNI"/'subcortical'
    brain = brainplots.Brain(id_= 'MNI', file_left=  file_left_hemisphere, file_right= file_right_hemisphere)
    #we only want to plot the insula therefore we set the opacity of the whole brain to 0
    brain.full.opacity = 0 
    path_cortical_regions = path_data/"MNI"/'label'/'gyri'

    #we load the insula outline
    brain.add_cortical_roi(["insula"], path_cortical_regions, 'rh',color=(0.8,0.8,0.8),opacity=0.3)  
    # Define colors
    colors=sns.color_palette("Dark2")
    #anterior insula is plotted in brown
    color_anterior = np.array(colors[-2])
    #posterior insula is plotted in gray
    color_posterior = np.array(colors[-1])       



    # Loop over participants adding all contacts to be plotted to contacts_list
    contacts_list=[]

    contacts = brainplots.Contacts("")  
            
    path_to_locations = path_data/"single_channelPressGammaprediction_results.csv"
    results=pd.read_csv(path_to_locations)
            # Set color and size 
    
    #initialize a name, xyz coordinate, color and size array for the contacts
    color_arr = np.zeros((len(results), 3), dtype=float)
    size_arr = np.zeros(len(results), dtype=int)  
    name_arr=[]
    xyz_arr=[]

                                    
                # Go through electrodes and check if they're in the anterior or posterior insula based on electrode labels
    for i in range(0,len(results)):
                    row=results.iloc[i]

                    #save in name_arr and xyz_arr the name and MNI coordinates of the contact
                    name_arr.append(row["channel_name"])
                    xyz_arr.append([row["MNI_x"],row["MNI_y"],row["MNI_z"]])
                        #set color based on is_anterior label
                    bool_val=row["is_anterior"]==1
                    if  bool_val:
                            color_arr[i] = color_anterior
                    else:
                            color_arr[i] = color_posterior
                        
                        #set size based on ttests t-values for decoding AUC. They are scaled using factor 2 and the squareroot
                        #of the t-value
                    size_arr[i] = 2*np.sqrt(max(0.0001,row["t_val"]))                               
                        

    #add the xyz coordinates to the contacts
    contacts.xyz=np.array(xyz_arr)
    #map everything to the left hemisphere
    contacts.xyz[:,0]=abs(contacts.xyz[:,0])
    #add the name array to the contacts
    contacts.names=np.array(name_arr)

            
    #in the end we append three more electrodes for size reference. They have the same size as t-values 1,2,4 and 8
    color_arr=np.append(color_arr,[(0.8,0.8,0.8),(0.8,0.8,0.8),(0.8,0.8,0.8),(0.8,0.8,0.8)],axis=0)
    size_arr=np.append(size_arr,[2*np.sqrt(1),2*np.sqrt(2),2*np.sqrt(4),2*np.sqrt(8)],axis=0)
    contacts.xyz=np.append(contacts.xyz,[[35,30,20],[35,20,20],[35,10,20],[35,0,20]],axis=0)
    contacts.names=np.append(contacts.names,["t1","t2","t3","t4"])
    #we add the color and size to the contacts
    contacts.add_color(color_arr)
    contacts.add_size(size_arr)
    #we append the contacts of this participant to the combined contacts_list
    contacts_list.append(contacts)
            
    #we plot the brainplot
    scene = brainplots.plot(brain, contacts=contacts_list, show=True)




    #Plot the distribution of the amount of channels and the amount of channels with t-value>2 over the anterior-inferior axis of the insula
    #we use y-coordinates in MNI space for the x-axis as they represent the anterior-posterior axis
    x = results["MNI_y"]
    #we also extract t-values from the results
    t_values = results["t_val"]

    # Create a grid to evaluate the KDE
    x_grid = np.linspace(min(x), max(x), 500)

    # compute the KDE for all electrodes
    kde_all = gaussian_kde(x,bw_method=0.3)
    density_all = kde_all(x_grid) * len(x) * (x_grid[1]-x_grid[0])  # scale to counts

    # compute the KDE for t > 2
    kde_t2 = gaussian_kde(x[t_values > 2],bw_method=0.3)
    density_t2 = kde_t2(x_grid) * np.sum(t_values > 2) * (x_grid[1]-x_grid[0])

    # Plot the distributions
    fig,ax=plt.subplots(figsize=(8,4))
    #distribution of all channels
    plt.fill_between(x_grid, 0, density_all, color='lightgray', label='All electrodes')
    #distribution of channels with t-value>2
    plt.fill_between(x_grid, 0, density_t2, color='#0000CC', alpha=0.7, label='t > 2')
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove all ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove axis labels (optional)
    ax.set_xlabel('')
    ax.set_ylabel('')


    plt.legend().set_visible(False)
    #save the figure
    fig.savefig(path_data/"plots"/"prediction_single_channel_insula_anterior_posterior.pdf", dpi=700)
    plt.show()
