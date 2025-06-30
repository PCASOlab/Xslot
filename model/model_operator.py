import torch
import torch.nn.functional as F
import numpy as np
from image_operator import basic_operator   
import cv2

def random_mask_out_dimension(tensor, mask_probability, dim):
    """
    Randomly masks out elements along the specified dimension of the tensor.

    Args:
    - tensor: input tensor of shape (B, C, D, H, W)
    - mask_probability: probability of masking out each element
    - dim: dimension along which to mask out elements (usually set to 2 for dimension D)

    Returns:
    - masked_tensor: tensor with elements randomly masked out along the specified dimension
    """
    # Determine the device of the input tensor
    device = tensor.device
    
    # Generate a random binary mask
    mask = torch.rand(tensor.size(dim), device=device) > mask_probability

    # Expand the mask to the shape of the input tensor along the specified dimension
    mask = mask.view(1, 1, -1, 1, 1).expand(tensor.size())

    # Apply the mask to the input tensor
    masked_tensor = tensor * mask

    return masked_tensor

def hide_patch(video_feature, patch_num=28*28, hide_prob=0.1, value=0,image_level= False):
    # assume patch_num is int**2
     
    bz,ch, D, H, W = video_feature.size()
    pn = int(patch_num ** (1/2))
    patch_size = int(W // pn)
    patch_offsets = [(x * patch_size, y * patch_size) for x in range(pn) for y in range(pn)]

    for d in range(D):
        for (px, py) in patch_offsets:
            if np.random.uniform() < hide_prob:

                video_feature[:,:, d, px:px + patch_size, py:py + patch_size] = value
    
    
    # for (px, py) in patch_offsets:
    #         if np.random.uniform() < hide_prob:
    #                 video_feature[:, :, :,px:px + patch_size, py:py + patch_size] = value # only deal train
   
    return video_feature



def Cam_mask_post_process(raw_masks,input,video_predict,multimask_output: bool = False):
        bz, ch, D, H, W = raw_masks.size()
        
        process_size =256
        raw_masks =   F.interpolate(raw_masks,  size=(D, process_size, process_size), mode='trilinear', align_corners=False)
        input =   F.interpolate(input,  size=(D, process_size, process_size), mode='trilinear', align_corners=False)

        bz_i, ch_i, D_i, H_i, W_i = input.size()

        bz, ch, D, H, W = raw_masks.size()
        video_predict = video_predict>0.5
        label_valid_repeat = video_predict.reshape(bz,ch,1,1,1).repeat(1,1,D,H,W)
        
        cam = (raw_masks>0.00)*raw_masks
        #channel wise 
        for i in range(ch):
            raw_masks[:,i,:,:,:] = cam[:,i,:,:,:] -torch.min(cam[:,i,:,:,:])
            mean = torch.sum ((raw_masks>0.0)* raw_masks)/ torch.sum (raw_masks>0.0)
            raw_masks[:,i,:,:,:] = raw_masks[:,i,:,:,:] /(torch.max(raw_masks[:,i,:,:,:] )+0.0000001)*4
        # raw_masks = raw_masks /(50+0.0000001) 
        raw_masks = raw_masks*label_valid_repeat
        # raw_masks = torch.clamp(raw_masks,0,1)    
        # mask_resample =   F.interpolate(raw_masks,  size=(D, H_i, W_i), mode='trilinear', align_corners=False)
        binary_mask =   raw_masks 
        # binary_mask = (self.mask_resample >0.05)*1.0

        # binary_mask =  self.mask_resample 

        # binary_mask = binary_mask.float(). to (self.device)
        # flattened_tensor = binary_mask.reshape(bz *ch* D,  256, 256)
        flattened_video= input.permute(0,2,1,3,4)
        flattened_video = flattened_video.reshape(bz * D, ch_i, H_i, W_i)

        

        flattened_mask= binary_mask.permute(0,2,1,3,4)
        flattened_mask = flattened_mask.reshape(bz * D, ch, H_i, W_i)



        output_mask = torch.zeros(bz * D, ch, 256, 256)
        
        
        post_process_mask = torch.zeros((bz * D, ch, H_i, W_i))
        with torch.no_grad():
                for i in range(ch):
                    for j in range (bz*D):
                        this_input_image=  flattened_video[j,:,:,:]

                        this_input_mask =  flattened_mask[j,i,:,:]
                        # this_input_mask =(this_input_mask >0.25) 

                        forground_num =  int(torch.sum(this_input_mask>0.2).item())
                        if forground_num>30:
                            this_input_mask= torch.tensor( post_process_softmask(this_input_mask.cpu().detach().numpy()  ,this_input_image.cpu().detach().numpy()  ))
                            # this_input_mask= torch.tensor( post_process_softmask2(this_input_mask))
                            binary_mask =  clear_boundary(binary_mask)

                            # this_input_mask= torch.tensor( post_process_softmask2(this_input_mask))


                            # this_input_mask =(this_input_mask>125)*1.0
                            post_process_mask[j,i,:,:] = this_input_mask
                        # coordinates = torch.ones(bz * D,1,2)*512.0
                        # coordinates= coordinates.cuda()
                        # labels = torch.ones(bz * D,1)
                        
                        


        # self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
         
        post_process_mask = post_process_mask.reshape (bz,D,ch,H_i,W_i).permute(0,2,1,3,4)
        # self.sam_mask = binary_mask
        return post_process_mask
        pass

    
def sam_mask_prompt_decode(sam_model,raw_masks,features,multimask_output: bool = False):
         
        bz, ch, D, H, W = raw_masks.size()
        bz_f, ch_f, D_f, H_f, W_f = features.size()

        raw_masks = raw_masks -torch.min(raw_masks)
        raw_masks = raw_masks /(torch.max(raw_masks)+0.0000001) 
        
        binary_mask = raw_masks 
        # binary_mask = (self.mask_resample >0.05)*1.0

        # binary_mask =  self.mask_resample 

        # binary_mask = binary_mask.float(). to (self.device)
       
        flattened_feature = features.permute(0,2,1,3,4)
        flattened_feature = flattened_feature.reshape(bz_f * D_f, ch_f, H_f, W_f)

        flattened_mask= binary_mask.permute(0,2,1,3,4)
        flattened_mask = flattened_mask.reshape(bz * D, ch, H, H)



        output_mask = torch.zeros(bz * D, ch, 256, 256)
        
        with torch.no_grad():
                for i in range(ch):
                    for j in range (bz*D):
                         

                        this_input_mask =  flattened_mask[j,i,:,:]
                        this_feature= flattened_feature[j:j+1,:,:,:]
                        this_input_mask =(this_input_mask >0.5)*1
                      
                        # this_input_mask =(this_input_mask>125)*1.0
                        
                        # coordinates = torch.ones(bz * D,1,2)*512.0
                        # coordinates= coordinates.cuda()
                        # labels = torch.ones(bz * D,1)
                        forground_num =  int(torch.sum(this_input_mask).item())
                        if forground_num>40:
                            foreground_indices = torch.nonzero(this_input_mask > 0.5, as_tuple=False)
                            # cntral = extract_central_point_coordinates(this_input_mask)
                                # Extract coordinates from indices
                            foreground_coordinates = foreground_indices[:, [1, 0]]  # Swap x, y to get (y, x) format
                            mask = decode_mask_with_multi_coord(sam_model,foreground_coordinates*1024/H,this_feature)
                            # mask = self.decode_mask_with_multi_coord(cntral[0]*1024/H_i,this_feature)

                            output_mask[j,i,:,:] = mask
                        


        # self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        sam_mask = output_mask.reshape (bz,D,ch,256,256).permute(0,2,1,3,4)
        # self.post_processed_masks = post_process_mask.reshape (bz,D,ch,H_i,W_i).permute(0,2,1,3,4)
        # self.sam_mask = binary_mask

        return sam_mask
     
def decode_mask_with_multi_coord(sam_model,foreground_coordinates,this_feature):
        N = foreground_coordinates.size(0)

# Calculate the step size
        step = N // 20

        # Sample coordinates using the step size
        if step == 0:
            sampled_coordinates= foreground_coordinates
        else:
            sampled_coordinates = foreground_coordinates[::step]
        # sampled_coordinates = foreground_coordinates 

        labels = torch.ones(1,1)
        # coordinates = cntral.view(1,1,2)*4
        # coordinates= coordinates.cuda()
        labels = labels.cuda()

        masks=[]
        N= len(sampled_coordinates)
        for i in range(len(sampled_coordinates)):
            coordinates = sampled_coordinates[i,:].view(1,1,2)
            coordinates= coordinates.cuda() 

            points = (coordinates, labels)

            # Embed prompts
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            # Predict masks
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings= this_feature,
                image_pe= sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            this_mask = (low_res_masks[0,0,:,:]>0)*1.0
            masks.append(this_mask)
        masks = torch.stack(masks)
        sum_mask = torch.sum(masks,dim=0)
        out_mask= (sum_mask>(2))*1.0
        # out_mask = this_mask
        return out_mask
     
     
    
        
def extract_central_point_coordinates(masks):
        boundary_size =10
        masks[:boundary_size, :] = 0
        masks[-boundary_size:, :] = 0
        masks[:, :boundary_size] = 0
        masks[:, -boundary_size:] = 0
        foreground_indices = torch.nonzero(masks > 0.5, as_tuple=False)

# Extract coordinates from indices
        foreground_coordinates = foreground_indices[:, [1, 0]]  # Swap x, y to get (y, x) format

        # Compute centroid of foreground coordinates
        centroid = torch.mean(foreground_coordinates.float(), dim=0)
        
        # Return centroid coordinates reshaped to [bz, 1, 2]
        return centroid.view(1, 1, 2)

def post_process_softmask(mask,image):
        
        
        image=  image.transpose(1,2,0)
        # mask = (mask>0.05)*mask
        # mask = mask -torch.min(mask)
        # mean = torch.sum ((mask>0.0)* mask)/ torch.sum (mask>0.0)
        # mask = mask /(mean+0.0000001)     
        # mask= mask*4
        # mask = (mask>0.3)*1.0
        mask =mask
        image= image
        # mask= np.clip(mask,0,1)
        mask=basic_operator .DCRF (image,mask)
        final_seg = np.argmax(mask, axis=0)
        # mask_uint8 = np.uint8(mask * 255)
        # # Ensure it's binary
        # _, mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # # Clear boundary errors if needed
        # # mask_cleaned = clear_boundary_errors(mask, boundary_size=5)

        # # Apply morphological opening if needed
        final_seg = apply_opening(final_seg.astype(np.uint8), kernel_size=5)
        return final_seg
def apply_opening(mask, kernel_size=3):
            """
            Apply opening operation to the mask.
            
            Parameters:
                mask (ndarray): Binary mask array.
                kernel_size (int): Size of the kernel for morphological opening.
            
            Returns:
                ndarray: Mask after applying morphological opening.
            """
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
def apply_closing(mask, kernel_size=3):
    """
    Apply closing operation to the mask.
    
    Parameters:
        mask (ndarray): Binary mask array.
        kernel_size (int): Size of the kernel for morphological closing.
    
    Returns:
        ndarray: Mask after applying morphological closing.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
def post_process_softmask2(mask ):
        
        
        
        # mask = mask -torch.min(mask)
        # mask = mask /(torch.max(mask)+0.0000001) 
        # mask= mask*4
        mask = (mask>0.3)*1.0
        mask =mask.cpu().detach().numpy()  
        
        mask_uint8 = np.uint8(mask * 255)
        # Ensure it's binary
        # _, mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Clear boundary errors if needed
        # mask_cleaned = clear_boundary_errors(mask, boundary_size=5)

        # Apply morphological opening if needed
        final_seg = apply_opening(mask_uint8.astype(np.uint8), kernel_size=5)
        final_seg = apply_closing(final_seg.astype(np.uint8), kernel_size=5)
        final_seg = apply_closing(final_seg.astype(np.uint8), kernel_size=5)

        final_seg = apply_opening(final_seg.astype(np.uint8), kernel_size=5)
        final_seg = apply_closing(final_seg.astype(np.uint8), kernel_size=3)
        final_seg = apply_closing(final_seg.astype(np.uint8), kernel_size=3)



        return final_seg

def CAM_to_slice_hardlabel(cam,video_predict):

        bz, ch, D, H, W = cam.size()
        raw_masks = (cam>0.0)*cam
        for i in range(ch):
            raw_masks[:,i,:,:,:] = cam[:,i,:,:,:] -torch.min(cam[:,i,:,:,:])
            mean = torch.sum ((raw_masks>0.0)* raw_masks)/ torch.sum (raw_masks>0.0)
            raw_masks[:,i,:,:,:] = raw_masks[:,i,:,:,:] /(torch.max(raw_masks[:,i,:,:,:] )+0.0000001)*4        
        binary_mask = (raw_masks >0.1)*1.0
        binary_mask =  clear_boundary(binary_mask)
        video_predict = video_predict>0.9
        label_valid_repeat = video_predict.reshape(bz,ch,1,1,1).repeat(1,1,D,H,W)
        binary_mask = binary_mask*label_valid_repeat
        # flatten_mask = binary_mask.view(bz,ch)
        count_masks = torch.sum(binary_mask, dim=(-1, -2), keepdim=True)
        slice_hard_label = (count_masks>6)*1.0
        return slice_hard_label,binary_mask
def clear_boundary(masks):
        boundary_size =1
        masks[:,:,:,:boundary_size, :] = 0
        masks[:,:,:,-boundary_size:, :] = 0
        masks[:,:,:,:, :boundary_size] = 0
        masks[:,:,:,:, -boundary_size:] = 0
        return masks