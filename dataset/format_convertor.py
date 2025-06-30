import numpy as np

class_name_Cholec_8k={0: 'Black Background',
                    1: 'Abdominal Wall',
                    2: 'Liver',
                    3: 'Gastrointestinal Tract',
                    4: 'Fat',
                    5: 'Grasper',
                    6: 'Connective Tissue',
                    7: 'Blood',
                    8: 'Cystic Duct',
                    9: 'L-hook Electrocautery',
                    10: 'Gallbladder',
                    11: 'Hepatic Vein',
                    12: 'Liver Ligament'}

categories = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6                  
    ]
categories_thoracic = [
    'Lymph node',
    'Vagus nereve',
    'Bronchus',
    'Lung parenchyma',
    'Instruments', 
    ]

categories_endovis =  [
    'Prograsp_Forceps_labels',
    'Large_Needle_Driver_labels',
    'Grasping_Retractor_labels',
    'Bipolar_Forceps_labels',
    'Vessel_Sealer_labels',
    'Monopolar_Curved_Scissors_labels',
    'Other_labels'
]
   
def label_from_endovis(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_endovis)
    new_label = inputlabel>5
    # new_label[:,0,:,:] = inputlabel[:,5,:,:]
    # new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>1)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
def label_from_Miccaitest(inputlabel):  #(13,29,256,256)
    in_ch, in_D, H, W = inputlabel.shape
    inputlabel = np.transpose(inputlabel, (1, 0, 2, 3))  # Transpose dimensions
    
    lenth = len(categories_endovis)
    
    # Create new_label while preserving NaN values
    new_label = np.where(np.isnan(inputlabel), np.nan, (inputlabel > 20) * 1.0)  # Use np.where to preserve NaN
    
    # Calculate frame_label, handle NaN values by checking if NaN is in the frame
    frame_label = np.sum(new_label, axis=(2, 3))
    frame_label = np.where(np.isnan(frame_label), np.nan, (frame_label > 1) * 1.0)
    
    # Calculate video_label, handle NaN values similarly
    video_label = np.nanmax(frame_label, axis=0)  # Use nanmax to ignore NaN values
    
    # Revert the label back to the original shape
    mask = np.transpose(new_label, (1, 0, 2, 3))
    
    return mask, frame_label, video_label

def label_from_seg8k_2_cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
    new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
     
def label_from_thoracic(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_thoracic)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,4,:,:] # swap
    # new_label[:,0,:,:] = inputlabel[:,5,:,:]
    # new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label


def label_from_thoracic2cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,0,:,:]
    new_label[:,1,:,:] = inputlabel[:,1,:,:]
    new_label[:,2,:,:] = inputlabel[:,2,:,:]
    new_label[:,3,:,:] = inputlabel[:,3,:,:]
    new_label[:,4,:,:] = inputlabel[:,4,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label



