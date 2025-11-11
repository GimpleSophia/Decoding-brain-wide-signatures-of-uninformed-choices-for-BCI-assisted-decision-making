## Seperate Insula contacts into Anterior and Posterior channels
# - finds insular channels 
# - based on their MNI labels and location in the insula determines if they should be labeled as anterior or posterior insula

#import some relevant packages
import nibabel as nib
import numpy as nppip 
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.io import loadmat
import pandas as pd
import scipy.io
from config import path_data

#specify some parameters
#is_whole_hemisphere can be selected to test the functions on the whole hemisphere
is_whole_hemisphere=0
#this plots the insula or hemisphere. In case of insula the anterior insula outline and the subregions are also plotted
is_plotting=1
#list of participants whose channels should be labeled
participant_list=["P01","P02","P03","P04","P05","P06"]

#participant specific file location of previous skript output
path_to_data=str(path_data)


#Paths to the MNI labels
#MNI coordinates
lh_faces_mat_path = path_to_data +r"\MNI\lh_pial_trivert.mat"
rh_faces_mat_path = path_to_data +r"\MNI\rh_pial_trivert.mat"

#insula label
lh_ins_label_path = path_to_data +r"\MNI\label\gyri\lh.insula.label"
rh_ins_label_path = path_to_data +r"\MNI\label\gyri\rh.insula.label"
#labels of insula subparts
#left hemisphere
lh_short_label_path = path_to_data +r"\MNI\label\destrieux\lh.G_Insular_short.label"
lh_sup_label_path = path_to_data +r"\MNI\label\destrieux\lh.S_circular_insula_sup.label"
lh_inf_label_path = path_to_data +r"\MNI\label\destrieux\lh.S_circular_insula_inf.label"
lh_ant_label_path = path_to_data +r"\MNI\label\destrieux\lh.S_circular_insula_ant.label"
lh_lg_label_path= path_to_data +r"\MNI\label\destrieux\lh.G_Ins_lg_and_S_cent_ins.label"
#right hemisphere
rh_short_label_path = path_to_data +r"\MNI\label\destrieux\rh.G_Insular_short.label"
rh_sup_label_path = path_to_data +r"\MNI\label\destrieux\rh.S_circular_insula_sup.label"
rh_inf_label_path = path_to_data +r"\MNI\label\destrieux\rh.S_circular_insula_inf.label"
rh_ant_label_path = path_to_data +r"\MNI\label\destrieux\rh.S_circular_insula_ant.label"
rh_lg_label_path= path_to_data +r"\MNI\label\destrieux\rh.G_Ins_lg_and_S_cent_ins.label"
# === Load surface coordinates ===
faces_data_rh = loadmat(rh_faces_mat_path)
coords_rh = faces_data_rh['vert']
faces_data_lh = loadmat(lh_faces_mat_path)
coords_lh = faces_data_lh['vert']

#load the coordinates of the insula region
rh_short_insula = nib.freesurfer.read_label(rh_short_label_path)
rh_short_coords = coords_rh[rh_short_insula]
lh_short_insula = nib.freesurfer.read_label(lh_short_label_path)
lh_short_coords = coords_lh[lh_short_insula]

#load the coordinates of all the insula subregions
rh_sup_insula = nib.freesurfer.read_label(rh_sup_label_path)
rh_sup_coords = coords_rh[rh_sup_insula]
lh_sup_insula = nib.freesurfer.read_label(lh_sup_label_path)
lh_sup_coords = coords_lh[lh_sup_insula]

rh_inf_insula = nib.freesurfer.read_label(rh_inf_label_path)
rh_inf_coords = coords_rh[rh_inf_insula]
lh_inf_insula = nib.freesurfer.read_label(lh_inf_label_path)
lh_inf_coords = coords_lh[lh_inf_insula]
rh_ant_insula = nib.freesurfer.read_label(rh_ant_label_path)
rh_ant_coords = coords_rh[rh_ant_insula]
lh_ant_insula = nib.freesurfer.read_label(lh_ant_label_path)
lh_ant_coords = coords_lh[lh_ant_insula]
rh_lg_insula = nib.freesurfer.read_label(rh_lg_label_path)
rh_lg_coords = coords_rh[rh_lg_insula]
lh_lg_insula = nib.freesurfer.read_label(lh_lg_label_path)
lh_lg_coords = coords_lh[lh_lg_insula]
rh_insula = nib.freesurfer.read_label(rh_ins_label_path)
rh_insula_coords = coords_rh[rh_insula]
lh_insula = nib.freesurfer.read_label(lh_ins_label_path)
lh_insula_coords = coords_lh[lh_insula]


def min_y_per_z_bin(points, n_bins=200, z_range=None):
    """
    This function computes for a cloud of points the minimum outline in y direction
    we use this to compute the posterior outline of the short gyri of the insula

    Parameters:
        points : np.ndarray, shape (N,3)
            Array of points with columns x, y, z.
        n_bins : int
            Number of bins along z-axis.
        z_range : tuple or None
            Optional (z_min, z_max) to define bin range. If None, use min/max of z.

    Returns:
        z_bin_centers : np.ndarray, shape (n_bins,)
            Center of each z-bin.
        min_y : np.ndarray, shape (n_bins,)
            Minimum y in each bin (NaN if no points in bin).
    """
    x, y, z = points[:,0], points[:,1], points[:,2]

    if z_range is None:
        z_min, z_max = z.min(), z.max()
    else:
        z_min, z_max = z_range

    # Define bin edges
    bin_edges = np.linspace(z_min, z_max, n_bins+1)
    z_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute minimum y in each bin
    min_y = np.empty(n_bins)
    min_y[:] = np.nan  # default NaN if bin is empty

    for i in range(n_bins):
        mask = (z >= bin_edges[i]) & (z < bin_edges[i+1])
        if np.any(mask):
            min_y[i] = np.min(y[mask])

    return z_bin_centers, min_y


def find_closest_z_index_and_y(z_centers,y_vals,coordinates):
    """
    This function takes coordinates of a 3D point p. It finds the closest
    z_value and the corresponding y-value. It is intended for comparing a datapoint
    to the posterior outline of the short gyri of the insula. It also compares if the
    y-value of point p is larger or smaller than the y-value of the outline and computes a label
    is_anterior


    Parameters
    ----------
    z_centers : np.ndarray
        Array of z-bin centers.
    y_vals : np.ndarray
        Array of correspoding y values.  
    coordinates : np.ndarray of length (at least) 3
        Coordinates of the datapoint to be evaluated

    Returns
    -------
    ind
        Index of the bin center closest to the z-coordinate of the datapoint.
    min_y
        corresponding minimal y-value.
    is_anterior
        label that determines if the point is anterior or not
    """
    ind=int(np.argmin(np.abs(z_centers - coordinates[2])))
    min_y=y_vals[ind]
    is_anterior=coordinates[1]>min_y
    return ind,min_y,is_anterior

if __name__=='__main__':
    
    #computes the posterior boundary of the short gyri of the insula for the left and right hemisphere
    points_rh = rh_short_coords # random example points
    rh_z_centers, rh_min_y_vals = min_y_per_z_bin(points_rh, n_bins=20)

    points_lh = lh_short_coords # random example points
    lh_z_centers, lh_min_y_vals = min_y_per_z_bin(points_lh, n_bins=20)

        
    #for all participants insular electrodes are labeled as 1 (anterior), 0 (posterior) or not in the insula (-1)
    for par in participant_list:
        #load channel information
        path=path_to_data+"/"+par+"/"
        channel_location_struct=pd.read_csv(path+'location_struct_'+par+'.csv')
        #fill the arrays is_ant_array and coordinate_array with the anterior label and the MNI coordinates
        is_ant_array=[]
        coordinate_array=[]
        for i in range(0,len(channel_location_struct)):
            row=channel_location_struct.iloc[i]
            chan_name=row["channel_name"]
            location=row["location"]
            coordinates_here=[row["x_MNI"],row["y_MNI"],row["z_MNI"]]
            coordinate_array.append(coordinates_here)

            #evaluate all insula channels
            if "ins" in location.lower():
                #short gyri and circular anterior channels are always labeled as anterior
                if "short" in location.lower() or "ant" in location.lower():
                    is_anterior=1
                #long gyri channels are always labeled as posterior
                elif "lg" in location.lower():
                    is_anterior=0
                #for all other channels we evaluate their location in relation to the posterior outline of the short gyri of the insula
                elif "lh" in location.lower():
                    ind,min_y,is_anterior=find_closest_z_index_and_y(lh_z_centers,lh_min_y_vals,coordinates_here)
                elif "rh" in location.lower():
                    ind,min_y,is_anterior=find_closest_z_index_and_y(rh_z_centers,rh_min_y_vals,coordinates_here)
                #we append 0 for posterior and a 1 for anterior
                if is_anterior>0:
                    is_ant_array.append(1)
                    print("anterior "+location)
                else:
                    is_ant_array.append(0)
                    print("posterior "+location)
            #non insula channels are always labeled with -1
            else:
                is_ant_array.append(-1)
        #we save the anterior label for each channel in an updated location struct
        coordinate_array=np.array(coordinate_array)
        channel_location_struct["is_anterior_insula"]=is_ant_array 
        channel_location_struct.to_csv(path+"output/"+'location_struct_insula_labels_'+par+'.csv')


        ## Plotting (optional)
        # - test the outline computation and labeling by plotting the hemisphere or insula and the outline.
        # - Subregions can be added additionally. 
        # - Channels are plotted in yellow if labeled ad anterior and in brown otherwise
        # - Color coding: orange (see-through)-insula, red- circular superior, blue-short gyri of the insula, green- inferior circular insula, yellow-long gyri of the insula
        if is_plotting:
            plt.figure()
            if is_whole_hemisphere:
                plt.scatter(coords_lh[:,1],coords_lh[:,2])
                plt.plot(lh_min_y_vals,lh_z_centers,color="black")
            else:
                #plt.scatter(rh_insula_coords[:,1],rh_insula_coords[:,2],color="orange",alpha=0.5)
                #plt.scatter(rh_sup_coords[:,1],rh_sup_coords[:,2],color="red",alpha=0.5)
                #plt.scatter(rh_inf_coords[:,1],rh_inf_coords[:,2],color="green",alpha=0.5)
                #plt.scatter(rh_ant_coords[:,1],rh_ant_coords[:,2],color="lightblue",alpha=0.5)
                #plt.scatter(rh_lg_coords[:,1],rh_lg_coords[:,2],color="yellow",alpha=0.5)
                #plt.scatter(rh_short_coords[:,1],rh_short_coords[:,2],color="blue",alpha=0.5)
                plt.plot(lh_min_y_vals,lh_z_centers,color="black")
                plt.scatter(lh_insula_coords[:,1],lh_insula_coords[:,2],color="orange",alpha=0.5)
                plt.scatter(lh_sup_coords[:,1],lh_sup_coords[:,2],color="red",alpha=0.5)
                plt.scatter(lh_inf_coords[:,1],lh_inf_coords[:,2],color="green",alpha=0.5)
                plt.scatter(lh_ant_coords[:,1],lh_ant_coords[:,2],color="lightblue",alpha=0.5)
                plt.scatter(lh_lg_coords[:,1],lh_lg_coords[:,2],color="yellow",alpha=0.5)
                plt.scatter(lh_short_coords[:,1],lh_short_coords[:,2],color="blue",alpha=0.5)
                coordinate_array=np.array(coordinate_array)
                is_ant_array=np.array(is_ant_array)
                ins_index=np.where(np.array(is_ant_array)>-1)[0]
                is_ant_plotting=is_ant_array[ins_index]
                coordinate_array_plotting=coordinate_array[ins_index]
                colors = np.where(is_ant_plotting, 'orange', 'brown')
                plt.scatter(coordinate_array_plotting[:,1],coordinate_array_plotting[:,2],c=colors)