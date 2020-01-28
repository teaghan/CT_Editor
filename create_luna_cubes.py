import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
from skimage import measure
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, erosion, binary_closing
from skimage.filters import roberts

from psopy import minimize

import SimpleITK as sitk
from glob import glob
import pandas as pd

def get_filename(file_list, case):      
    for f in file_list:
        if case in f:
            return(f)

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    
    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    # Read direction
    direction = itkimage.GetDirection()
    direction = np.array([direction[8], direction[4], direction[0]])
    
    return ct_scan, origin, spacing, direction

'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    return np.rint((world_coordinates - origin)/spacing)

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def nearest_even(x):
    # Add epsilon to avoid issues with rounding .5 down
    eps = 1e-6
    return np.rint((x+eps)/2.)*2

def resize_image(image, orig_spacing, new_spacing=(1.95, 0.65, 0.65)):
    # Calculate resize factor
    resize_factor = orig_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / image.shape
    new_spacing = orig_spacing / real_resize
    return scipy.ndimage.interpolation.zoom(image, real_resize, mode='nearest'), new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def get_segmented_lungs(im, cutoff=0, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(6, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < cutoff
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    return binary.astype(int)
    
def segment_lung_mask(ct_scan, cutoff=-350):
    return np.asarray([get_segmented_lungs(slice_, cutoff=-400) for slice_ in ct_scan])

def segment_opt(x0, segmented_lungs, ct_array, box_size, lung_frac_tgt, avg_int_tgt, int_lw, frac_lw):
    segmented_cube = segmented_lungs[int(x0[0]):int(x0[0])+box_size[0],
                           int(x0[1]):int(x0[1])+box_size[1],
                           int(x0[2]):int(x0[2])+box_size[2]]
    ct_cube = ct_array[int(x0[0]):int(x0[0])+box_size[0],
                           int(x0[1]):int(x0[1])+box_size[1],
                           int(x0[2]):int(x0[2])+box_size[2]]
    lung_frac = np.sum(segmented_cube) / np.prod(box_size)
    avg_int = np.mean(ct_cube)
    
    frac_loss = (lung_frac-lung_frac_tgt)**2
    int_loss = (avg_int-avg_int_tgt)**2
    
    return frac_lw*frac_loss + int_lw*int_loss

def select_nontumor_cube(ct_array, segmented_lungs, box_size, n_particles=50, 
                         lung_frac_tgt=1., avg_int_tgt=-700, int_lw=1e-5, frac_lw=1):
    # Use Particle Swarm Optimization (PSO) to locate a cube that contains mostly the lungs
    # but does not contain the cancerous cube.

    # A "swarm" of initial guesses for the starting corner of the cube
    x0 = np.hstack((np.random.uniform(0, (segmented_lungs.shape[0]-box_size[0]), (n_particles, 1)),
                    np.random.uniform(0, (segmented_lungs.shape[1]-box_size[1]), (n_particles, 1)),
                    np.random.uniform(0, (segmented_lungs.shape[2]-box_size[2]), (n_particles, 1))))
    
    # The corner must be within the limits of the original scan
    constraints = ({'type': 'ineq', 'fun': lambda x: x[0]},
                   {'type': 'ineq', 'fun': lambda x: x[1]},
                   {'type': 'ineq', 'fun': lambda x: x[2]},
                   {'type': 'stin', 'fun': lambda x: (segmented_lungs.shape[0]-box_size[0]) - x[0]},
                   {'type': 'stin', 'fun': lambda x: (segmented_lungs.shape[1]-box_size[1]) - x[1]},
                   {'type': 'stin', 'fun': lambda x: (segmented_lungs.shape[2]-box_size[2]) - x[2]})
    
    # Optimize the corner location
    res = minimize(segment_opt, x0, constraints=constraints, args=(segmented_lungs, ct_array, box_size, 
                                                                   lung_frac_tgt, avg_int_tgt, 
                                                                   int_lw, frac_lw))

    box_start_good = np.round(res.x).astype(int)
    
    segmented_cube = segmented_lungs[box_start_good[0]:box_start_good[0]+box_size[0], 
                                     box_start_good[1]:box_start_good[1]+box_size[1],
                                     box_start_good[2]:box_start_good[2]+box_size[2]]
    lung_frac = np.sum(segmented_cube) / np.prod(box_size)
    print('\t\t%0.f%% of the selected cube resides within the lungs.'%(lung_frac*100))

    # Segment this cube
    good_segment = ct_array[box_start_good[0]:box_start_good[0]+box_size[0], 
                            box_start_good[1]:box_start_good[1]+box_size[1],
                            box_start_good[2]:box_start_good[2]+box_size[2]]
    avg_int = np.mean(good_segment)
    
    print('\t\tThe average intensity of the cube is %0.2f HU.'%(avg_int))
    
    # Mask this area to not use it again
    segmented_lungs[box_start_good[0]:box_start_good[0]+box_size[0], 
                    box_start_good[1]:box_start_good[1]+box_size[1],
                    box_start_good[2]:box_start_good[2]+box_size[2]] = -1000
    
    return good_segment, segmented_lungs, box_start_good, lung_frac, avg_int

    
def main():
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'data/')
    
    dcim_paths = ['LUNA16/subset0/', 'LUNA16/subset1/', 'LUNA16/subset2/', 
                  'LUNA16/subset3/', 'LUNA16/subset4/', 'LUNA16/subset5/', 
                  'LUNA16/subset6/', 'LUNA16/subset7/', 'LUNA16/subset8/',
                  'LUNA16/subset9/']    
    annotation_file = os.path.join(data_dir, 'LUNA16/annotations_enhanced.csv')
    
    # Uniform distributions to select lung perc from [min, max]
    lung_frac_lims = [[0.4,0.6], [0.6,0.95], [0.99, 1.]]
    # Normal distributions to select avg intensity from [avg, std]
    avg_int_dist = [[-600, 100], [-700, 100], [-800, 100]]
    # Number of healthy sets
    num_sets = 6

    # CT files
    file_list = []
    for dcim_path in dcim_paths:
        file_list = file_list + glob(os.path.join(data_dir, dcim_path)+'*.mhd')

    # Create data files
    savename_nod = os.path.join(data_dir, 'nodule_segments.h5')
    dt_scan = h5py.special_dtype(vlen=np.dtype('float64'))
    dt_str = h5py.special_dtype(vlen=bytes)
    if not os.path.exists(savename_nod):
        with h5py.File(savename_nod, "w") as f:    
            # Datasets for h5 file
            segments_ds = f.create_dataset('Segment', (1,), maxshape=(None,), dtype=dt_scan)
            resolution_ds = f.create_dataset('Resolution', (1,3), maxshape=(None,3), dtype='f')
            pat_id_ds = f.create_dataset('Patient ID', (1,), maxshape=(None,), dtype=dt_str)
            location_ds = f.create_dataset('Location', (1,3), maxshape=(None,3), dtype='f')
            shape_ds = f.create_dataset('Shape', (1,3), maxshape=(None,3), dtype="i")
            diameter_ds = f.create_dataset('Diameter', (1,), maxshape=(None,), dtype='f')
            calcification_ds = f.create_dataset('Calcification', (1,6), maxshape=(None,6), dtype='f')
            sphericity_ds = f.create_dataset('Sphericity', (1,3), maxshape=(None,3), dtype='f')
            lobulation_ds = f.create_dataset('Lobulation', (1,), maxshape=(None,), dtype='f')
            spiculation_ds = f.create_dataset('Spiculation', (1,), maxshape=(None,), dtype='f')
            texture_ds = f.create_dataset('Texture', (1,3), maxshape=(None,3), dtype='f')
            malignancy_ds = f.create_dataset('Malignancy', (1,), maxshape=(None,), dtype='f')
            
            pat_ids_complete = []
        first_nod_entry = True
    else:
        with h5py.File(savename_nod, "r") as f:    
            pat_ids_complete = list(f['Patient ID'][:].astype("str"))
        first_nod_entry = False


    savename_hlt = os.path.join(data_dir, 'healthy_segments.h5')
    if not os.path.exists(savename_hlt):
        with h5py.File(savename_hlt, "w") as f:    

            # Datasets for h5 file
            segments_ds = f.create_dataset('Segment', (1,), maxshape=(None,), dtype=dt_scan)
            resolution_ds = f.create_dataset('Resolution', (1,3), maxshape=(None,3), dtype='f')
            pat_id_ds = f.create_dataset('Patient ID', (1,), maxshape=(None,), dtype=dt_str)
            location_ds = f.create_dataset('Location', (1,3), maxshape=(None,3), dtype='f')
            shape_ds = f.create_dataset('Shape', (1,3), maxshape=(None,3), dtype="i")
            lung_frac_ds = f.create_dataset('Lung Fraction', (1,), maxshape=(None,), dtype='f')
        first_hlt_entry = True
    else:
        first_hlt_entry = False

    # The locations and annotations for each of the nodes
    df_node = pd.read_csv(annotation_file)

    # Loop through patients
    for i, img_file in enumerate(file_list):

        # Unique patient ID
        pat_id = str(img_file).split('/')[-1][:-4]

        # Get all nodules associate with file
        mini_df = df_node[df_node['seriesuid']==img_file.split('/')[-1][:-4]]

        print('Patient (%i/%i): %s' % (i, len(file_list), pat_id))
        if pat_id in pat_ids_complete:
            continue

        # Skip patients without a nodule
        if len(mini_df)==0:
            continue

        # Load CT data
        ct_scan, origin, orig_spacing, direction = load_itk(img_file)

        # Check if bad file
        if len(np.where(ct_scan==0)[0])/np.prod(ct_scan.shape)>0.6:
            continue

        # Rescale the CT scan to a common resolution
        ct_scan, new_spacing = resize_image(ct_scan, orig_spacing)

        # Segment lungs (1=lung, 0=not lung)
        segmented_lungs = segment_lung_mask(ct_scan, False)

        # Iterate through nodules
        box_sizes = []
        print('\tFound %i nodule(s).' % len(mini_df))
        for node_idx, cur_row in mini_df.iterrows(): 
            # Centre location of nodule in world coordinates
            world_loc = np.array([cur_row['coordZ'], cur_row['coordY'], cur_row['coordX']])
            # Centre location of nodule in voxel coordinates
            vox_loc = world_2_voxel(world_loc, origin, orig_spacing)
            vox_loc *= direction
            # Rescale this location in the same way we did the CT scan
            vox_loc = np.rint(vox_loc * (orig_spacing / new_spacing)).astype(int)

            # Annotations of nodule
            diameter = cur_row['diameter_mm']
            calcification = np.array(cur_row['calcification'].replace(']','').replace('[','').split()).astype(float)
            sphericity = np.array(cur_row['sphericity'].replace(']','').replace('[','').split()).astype(float)
            lobulation = cur_row['lobulation']
            spiculation = cur_row['spiculation']
            texture = np.array(cur_row['texture'].replace(']','').replace('[','').split()).astype(float)
            malignancy = cur_row['malignancy']

            # Compute the z side length of the box after adding a 4 mm to each side.
            # We force this to be an even number of pixels so that the cube is centered around the nodule.
            side_len_z = nearest_even((diameter+8)/new_spacing[0])
            # Compute the other side lengths
            aspect = np.rint(new_spacing[0]/new_spacing[1])
            box_size = np.array([side_len_z, side_len_z*aspect, side_len_z*aspect]).astype(int)
            print('\t\tCreating nodule cube of size (%i, %i, %i) pixels...' % (box_size[0], 
                                                                               box_size[1], 
                                                                               box_size[2]))
            box_sizes.append(box_size)

            # Select the nodule from the CT scan
            start_indx = (vox_loc-(box_size/2)).astype(int)
            nodule_cube = ct_scan[start_indx[0]:start_indx[0]+box_size[0],
                                  start_indx[1]:start_indx[1]+box_size[1],
                                  start_indx[2]:start_indx[2]+box_size[2]]

            # Save data
            with h5py.File(savename_nod, "r+") as f:    
                if not first_nod_entry:
                    f['Resolution'].resize(f['Resolution'].shape[0]+1, axis=0)
                    f['Patient ID'].resize(f['Patient ID'].shape[0]+1, axis=0)
                    f['Location'].resize(f['Location'].shape[0]+1, axis=0)
                    f['Diameter'].resize(f['Diameter'].shape[0]+1, axis=0)
                    f['Calcification'].resize(f['Calcification'].shape[0]+1, axis=0)
                    f['Sphericity'].resize(f['Sphericity'].shape[0]+1, axis=0)
                    f['Lobulation'].resize(f['Lobulation'].shape[0]+1, axis=0)
                    f['Spiculation'].resize(f['Spiculation'].shape[0]+1, axis=0)
                    f['Texture'].resize(f['Texture'].shape[0]+1, axis=0)
                    f['Malignancy'].resize(f['Malignancy'].shape[0]+1, axis=0)
                    f['Shape'].resize(f['Shape'].shape[0]+1, axis=0)
                    f['Segment'].resize(f['Segment'].shape[0]+1, axis=0)
                f['Resolution'][-1] = new_spacing
                f['Patient ID'][-1] = pat_id
                f['Location'][-1] = world_loc
                f['Diameter'][-1] = diameter                
                f['Calcification'][-1] = calcification
                f['Sphericity'][-1] = sphericity
                f['Lobulation'][-1] = lobulation
                f['Spiculation'][-1] = spiculation
                f['Texture'][-1] = texture
                f['Malignancy'][-1] = malignancy
                f['Shape'][-1] = nodule_cube.shape
                f['Segment'][-1] = nodule_cube.flatten()
                first_nod_entry = False

            # Mask tumor voxels as -1000 to give large "loss" when looking for "good" region
            segmented_lungs[start_indx[0]:start_indx[0]+box_size[0],
                            start_indx[1]:start_indx[1]+box_size[1],
                            start_indx[2]:start_indx[2]+box_size[2]] = -1000

        print('\tCreating %i equivalently size "non-cancerous" cubes for each...'%(3*num_sets))
        for jj in range(num_sets):
            # Find random cubes within lungs but outside of above tumor regions
            for box_size in box_sizes:
                for ii in range(len(lung_frac_lims)):
                    # Target lung fraction
                    lung_frac_tgt = np.random.uniform(lung_frac_lims[ii][0],
                                                      lung_frac_lims[ii][1])
                    # Target avg intensity
                    avg_int_tgt = np.random.normal(avg_int_dist[ii][0], avg_int_dist[ii][1])

                    print('\t\tTargets: %0.f%% and %0.2f HU ' % (100*lung_frac_tgt, avg_int_tgt))

                    # Find random cubes within lungs but outside of above tumor regions
                    (good_segment, segmented_lungs, 
                     box_start_good, lung_frac, avg_int) = select_nontumor_cube(ct_scan, 
                                                                                segmented_lungs, 
                                                                                box_size,
                                                                                lung_frac_tgt=lung_frac_tgt, 
                                                                                avg_int_tgt=avg_int_tgt)

                    if lung_frac<=0:
                        print('\t\t\tBad segment, skipping...')
                        continue

                    # Find centre location of healthy segment
                    healthy_vox_loc = box_start_good+box_size/2
                    # Undo scaling
                    healthy_vox_loc = healthy_vox_loc * (new_spacing / orig_spacing)
                    # Flip indexing depending on "direction" of scan
                    healthy_vox_loc *= direction
                    # Return to world coordinates
                    healthy_world_loc = voxel_2_world(healthy_vox_loc, origin, orig_spacing)
                    # Save data
                    if good_segment is not None:
                        with h5py.File(savename_hlt, "r+") as f:    
                            if not first_hlt_entry:
                                f['Resolution'].resize(f['Resolution'].shape[0]+1, axis=0)
                                f['Patient ID'].resize(f['Patient ID'].shape[0]+1, axis=0)
                                f['Location'].resize(f['Location'].shape[0]+1, axis=0)
                                f['Shape'].resize(f['Shape'].shape[0]+1, axis=0)
                                f['Segment'].resize(f['Segment'].shape[0]+1, axis=0)
                                f['Lung Fraction'].resize(f['Lung Fraction'].shape[0]+1, axis=0)
                            f['Resolution'][-1] = new_spacing
                            f['Patient ID'][-1] = pat_id
                            f['Location'][-1] = healthy_world_loc
                            f['Shape'][-1] = good_segment.shape
                            f['Segment'][-1] = good_segment.flatten()
                            f['Lung Fraction'][-1] = lung_frac
                            first_hlt_entry = False
                    else:
                        print('\t\tNo good segment found.')
        pat_ids_complete.append(pat_id)
    
if __name__== "__main__":
    main()