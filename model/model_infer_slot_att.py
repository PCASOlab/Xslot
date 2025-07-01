import torch
import torch.nn as nn
import argparse
import logging
import warnings
from typing import Any, Dict, Optional
import os
# import pytorch_lightning as pl

import torch.nn.functional as F
import pathlib
# from pytorch_lightning.utilities import rank_zero_info as log_info

# import torchvision.models as models
import cv2
 
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_student,Display_final_SAM
# from working_dir_root import Enable_teacher
from dataset.dataset import class_weights
import numpy as np
from image_operator import basic_operator   
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax
# from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import Enable_student,Random_mask_temporal_feature,Random_mask_patch_feature,Display_fuse_TC_ST
from working_dir_root import Use_max_error_rejection,Evaluation_slots,Display_embedding
from model import model_operator, models
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset.dataset import label_mask,Mask_out_partial_label,Obj_num
from video_SA. videosaur import configuration, data, metrics, utils
from working_dir_root import Display_flag,video_saur_pretrain,Evaluation_slots
import random
import model.model_operator_slots as slots_op
import model.display. model_vis as modelVis

if Evaluation == True:
    learningR=0
    Weight_decay=0
# learningR = 0.0001
def select_gpus(gpu_selection):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs available:", num_gpus)
        if gpu_selection == "all":
            device = torch.device("cuda" if num_gpus > 0 else "cpu")
            # if num_gpus > 1:
            #     device = torch.device("cuda:0," + ",".join([str(i) for i in range(1, num_gpus)]))
        elif gpu_selection.isdigit():
            gpu_index = int(gpu_selection)
            device = torch.device("cuda:" + gpu_selection if gpu_index < num_gpus else "cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device
class _Model_infer(object):
    def __init__(self,args=None, GPU_mode =True,num_gpus=1,Using_contrast=False,Using_SP_regu = False, Using_SP = False,
                 Using_slot_bert=False,slot_ini = "rnn",Sim_threshold=0.9,Mask_feat=False,cTemp=1.0,img_sim=False,
                 gpu_selection = "all",pooling="rank",TPC=True):
        config_overrides=None
        if GPU_mode ==True:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = select_gpus(gpu_selection)
        else:
            device = torch.device("cpu")
        self.device = device
        self.use_contrast = Using_contrast
        self.Using_SP_regu =  Using_SP_regu
        self.Using_SP  = Using_SP
        self.use_bert = Using_slot_bert
        self.slot_ini = slot_ini
        self.Mask_feat= Mask_feat
        self.cTemp = cTemp
        self.img_sim = img_sim
        if Evaluation_slots:
            self.Mask_feat = False
        rank_zero = utils.get_rank() == 0
        if config_overrides is None and args is not None:
            config_overrides = args.config_overrides
        if args is not None:
            config = configuration.load_config(args.config, config_overrides)
        if args.config_overrides_file is not None:
            config = configuration.override_config(
                config,
                override_config_path=args.config_overrides_file,
                additional_overrides=config_overrides,
            )

        

        if config.train_metrics is not None:
            train_metrics = {
                name: metrics.build(config) for name, config in config.train_metrics.items()
            }
        else:
            train_metrics = None

        if config.val_metrics is not None:
            val_metrics = {name: metrics.build(config) for name, config in config.val_metrics.items()}
        else:
            val_metrics = None

        self.model = models.build(config.model, config.optimizer, train_metrics, val_metrics,Using_SP=self.Using_SP,Sim_threshold=Sim_threshold)
       
       
        optimizers = self.model .configure_optimizers()
        # If `configure_optimizers` returns a dictionary
        if isinstance(optimizers, dict):
            optimizer = optimizers['optimizer']
        elif isinstance(optimizers, (list, tuple)):
            optimizer = optimizers[0]  # assuming there's at least one optimizer
        else:
            optimizer = optimizers
        # self.optimizer = optimizer
        if self.Using_SP == True:
            self.optimizer = torch.optim.AdamW ([ 
            # {'params': self.model.initializer.parameters(),'lr': learningR},
            # {'params': self.model.encoder.module.output_transform.parameters(),'lr': learningR},
            # {'params': self.model.processor.parameters(),'lr': learningR},
            {'params': self.model.presence_nn.parameters(),'lr': learningR},
            {'params': self.model.decoder.parameters(),'lr': learningR},
            {'params': self.model.temporal_binder.parameters(),'lr': learningR},
            {'params': self.model.future_state_prdt.parameters(),'lr': learningR}
            ], weight_decay=Weight_decay)
        else:
            self.optimizer = torch.optim.AdamW ([ 
            {'params': self.model.initializer.parameters(),'lr': learningR},
            # {'params': self.model.encoder.module.output_transform.parameters(),'lr': learningR},
            {'params': self.model.processor.parameters(),'lr': learningR},
            {'params': self.model.presence_nn.parameters(),'lr': learningR},
            {'params': self.model.decoder.parameters(),'lr': learningR},
            {'params': self.model.temporal_binder.parameters(),'lr': learningR}
            ], weight_decay=Weight_decay)
        self.input_size = 224
        self.inter_bz = 1
        # self.model.train(True)

        # if GPU_mode == True:
        #     if num_gpus > 1 and gpu_selection == "all":
        #         # self.VideoNets.classifier = torch.nn.DataParallel(self.VideoNets.classifier)
        #         # self.VideoNets.blocks = torch.nn.DataParallel(self.VideoNets.blocks)
        #         self.model = torch.nn.DataParallel(self.model)
        self.model.to (device)
        # self.model.train(True)
        if Evaluation_slots == False:
            self.model.train(True)
        else:
            self.model.train(False)
        # self.model.initializer.n_slots = config.n_slots

    #     output_dummy = model(dummy_batch)
    # # Create the dictionary and assign the tensor to the "video" key
    #     loss = model.compute_loss(output_dummy)
    #     loss[0].backward()
    #     optimizer.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def convert_mask(self, input, outputs, softmasks=True):

        bz, D, ch, H, W = input["video"].size()

        b, f, n_slots, hw = outputs["decoder"]["masks"].shape
        h = int(np.sqrt(hw))
        w = h
        masks_video = outputs["decoder"]["masks"].reshape(b, f, n_slots, h, w)
        masks_video = masks_video.permute(0,2,1,3,4)
        masks_video = F.interpolate(masks_video,  size=(D, H, W), 
                                      mode='trilinear', align_corners=False)
        if not softmasks:
            ind = torch.argmax(masks_video, dim=1, keepdim=True)
            masks_video = torch.zeros_like(masks_video)
            masks_video.scatter_(1, ind, 1)
        
        return masks_video
    def orthogonality_loss(self,batch_frames_matrix):
    # Reshape the matrix from (Batch, Frames, M, N) to (Batch*Frames, M, N)
        batch_size, num_frames, M, N = batch_frames_matrix.size()
        reshaped_matrix = batch_frames_matrix.reshape(batch_size * num_frames, M, N)
        
        # Normalize each row (vector) in the matrix
        normalized_matrix = reshaped_matrix / reshaped_matrix.norm(dim=2, keepdim=True)
        
        # Compute the cosine similarity matrix for each (M, N) matrix
        # (Batch*Frames, M, M): Cosine similarity between M vectors in each matrix
        cosine_sim_matrix = torch.bmm(normalized_matrix, normalized_matrix.transpose(1, 2))
        
        # Create an identity matrix to exclude diagonal elements (self-similarity)
        identity_matrix = torch.eye(M).to(batch_frames_matrix.device)
        
        # Penalize the off-diagonal elements (cosine similarity between different vectors)
        # We subtract the identity matrix to zero-out diagonal elements, and take absolute value of off-diagonals
        off_diagonal_loss = cosine_sim_matrix - identity_matrix[None, :, :]  # Broadcast identity over batch
        
        # Sum of absolute values of off-diagonal elements
        loss = off_diagonal_loss.abs().sum()
        
        return loss
    def slot_contrastive_loss(self, batch_frames_matrix, tau=0.1):
        """
        Computes the contrastive loss for slot orthogonality using log and exp.
        All slots are treated as negative pairs to encourage dissimilarity (orthogonality).
        
        Args:
            batch_frames_matrix (torch.Tensor): Tensor of shape (Batch, Frames, K, d_slot),
                                                where K is the number of slots.
            tau (float): Temperature parameter for scaling cosine similarity.
        
        Returns:
            loss (torch.Tensor): Contrastive loss scalar.
        """
        batch_size, num_frames, K, d_slot = batch_frames_matrix.size()
        
        # Reshape the matrix to (Batch*Frames, K, d_slot)
        reshaped_matrix = batch_frames_matrix.view(batch_size * num_frames, K, d_slot)
        
        # Normalize each row (vector) in the matrix to unit norm
        normalized_matrix = F.normalize(reshaped_matrix, dim=-1)  # Shape: (Batch*Frames, K, d_slot)
        
        # Compute cosine similarity matrix for all slots within each frame
        similarity_matrix = torch.bmm(normalized_matrix, normalized_matrix.transpose(1, 2))  # (Batch*Frames, K, K)
        
        # Exclude self-similarity by subtracting the identity matrix
        identity_matrix = torch.eye(K, device=batch_frames_matrix.device).expand_as(similarity_matrix)
        similarity_matrix_no_diag = similarity_matrix - identity_matrix

        # Compute the contrastive loss using log and exp (with a negative sign for dissimilarity)
        exp_sim = torch.exp(-similarity_matrix_no_diag / tau)  # Exponential similarity (scaled)
        denominator = exp_sim.sum(dim=-1, keepdim=True)  # Sum across slots (excluding self-similarity)

        # Calculate loss as negative log of similarity normalized by the sum
        loss_matrix = -torch.log(exp_sim / denominator)  # Contrastive loss for each slot pair
        contrastive_loss = loss_matrix.mean()  # Average across all slots and frames
        
        return contrastive_loss
    def orthogonality_loss2(self, batch_frames_matrix, temperature=1.0):
        # Reshape the matrix from (Batch, Frames, M, N) to (Batch*Frames, M, N)
        batch_size, num_frames, M, N = batch_frames_matrix.size()
        reshaped_matrix = batch_frames_matrix.reshape(batch_size * num_frames, M, N)
        
        # Normalize each row (vector) in the matrix
        normalized_matrix = reshaped_matrix / reshaped_matrix.norm(dim=2, keepdim=True)
        
        # Compute the cosine similarity matrix for each (M, N) matrix
        # (Batch*Frames, M, M): Cosine similarity between M vectors in each matrix
        cosine_sim_matrix = torch.bmm(normalized_matrix, normalized_matrix.transpose(1, 2))
        
        # Apply temperature scaling to the cosine similarity
        cosine_sim_matrix = cosine_sim_matrix / temperature
        
        # Create an identity matrix to exclude diagonal elements (self-similarity)
        identity_matrix = torch.eye(M, device=batch_frames_matrix.device).expand_as(cosine_sim_matrix)
        
        # Mask the diagonal elements (self-similarity)
        # cosine_sim_matrix = cosine_sim_matrix - identity_matrix[None, :, :]  # Broadcast identity over batch
        
        # Apply softmax to each row (cross-entropy-like calculation)
        # Convert the cosine similarities into probabilities
        # cosine_sim_matrix_exp = torch.exp(cosine_sim_matrix)
        # cosine_sim_matrix_prob = cosine_sim_matrix_exp / cosine_sim_matrix_exp.sum(dim=2, keepdim=True)
        
        # Calculate the negative log likelihood (cross-entropy-like loss)
        # For orthogonality, we want all off-diagonal elements to be low, so we focus on the cross-entropy-like term.
        # Here we assume that we want to "penalize" the model for having high values in off-diagonal elements.
        # cross_entropy_loss = -torch.log(cosine_sim_matrix_prob + 1e-8).mean()
        cross_entropy_loss = F.cross_entropy(cosine_sim_matrix,identity_matrix)
    
        return cross_entropy_loss
    def SlotMask_regulation_loss(self, slot_keep: torch.Tensor, linear_weight = 0.1,
        quadratic_weight=  0.1,
        quadratic_bias = 0.5) :
         
        # Calculate the mean sparsity degree across all frames and features
        sparse_degree = torch.mean(slot_keep)  # Averaged over batch, T, and N

     

        # Sparse penalty calculation
        loss = linear_weight * sparse_degree + quadratic_weight * (sparse_degree - quadratic_bias) ** 2
        return loss
    def forward(self,input,input_flows, features,Enable_student,epoch=0):
        # self.res_f = self.resnet(input)

        bz, ch, D, H, W = input.size()
        activationLU = nn.ReLU()


        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        self.input_resample= (self.input_resample-124.0)/60.0
        video_input = {"video":  self.input_resample.permute(0,2,1,3,4)}
        
        self.output = self.model(video_input,self.use_bert, slot_ini=self.slot_ini,Mask_feat=self.Mask_feat,img_sim = self.img_sim,epoch=epoch)
        self.cam3D = self.convert_mask(video_input,self.output)
        bz, ch_n, D, H, W = self.cam3D.size()

        self.raw_cam = self.cam3D
        self.final_output = torch.ones(bz,ch_n,1,1,1)
        self.gradcam = None
        self.direct_frame_output = None

        if Display_embedding:
                slots = self.output['processor']["state"] 
                modelVis.plot_all_frames_on_projection_vizdom_2d_tsne(slots)
                modelVis.plot_all_frames_on_hypersphere_vizdom_tsne(slots)
                


    
    def optimization(self, label,Enable_student):
        # for param_group in  self.optimizer.param_groups:
            self.optimizer.zero_grad()
            # self.set_requires_grad(self.VideoNets, False)
            slots = self.output['processor']["state"] 
            loss_ortho = self.orthogonality_loss2(slots,temperature=self.cTemp)
            # loss_affin = slots_op.affinity_matrix_regularization(slots)
            loss = self.model.compute_loss(self.output)
            # final_loss = loss[0]  
             
            if self.Using_SP_regu == False:
                loss_presence_p = 0
                self. lossDisplay_p = torch.tensor(0).cuda()

            else:
                loss_presence_p = self.SlotMask_regulation_loss(slot_keep=self.output["presence_p"])
                self. lossDisplay_p = loss_presence_p


            if self.use_contrast == False:
                final_loss = loss[0] +  loss_presence_p#+ 0.00001*loss_ortho#+ 0.000001 * loss_affin
                
            else:
                final_loss = loss[0] +  loss_presence_p+ 0.001*loss_ortho#+ 0.000001 * loss_affin


            final_loss.backward()
            self.optimizer.step()
            self.lossDisplay = loss[0]
            # self. lossDisplay_p = loss[1]['loss_timesim']
            self. lossDisplay_s = loss_ortho


 
