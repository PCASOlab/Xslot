# from working_para.working_dir_root_train import *
# from working_para.working_dir_root_eval import *
# from working_para.working_dir_root_train_miccai import *
# from working_para.working_dir_root_train_Thoracic import *
# from main.sbert.MLP_ds_simmerger_FL_iteration import Traning_set
# from working_para.working_dir_root_train_cholec import *

# from working_para.working_dir_root_eval_p1 import *
# working_dir.py
import os
Machine_ID = 3
# Set default mode if not specified
import_mode = os.environ.get('WORKING_DIR_IMPORT_MODE', 'train_cholec')
if Machine_ID == 3:
    # Dynamically import based on the mode
    if import_mode == 'train_cholec':
        from working_para.working_dir_root_train_cholec_p3 import *
    elif import_mode == 'eval_cholec':
        from working_para.working_dir_root_eval_cholec_p3 import *
    elif import_mode == 'train_miccai':
        from working_para.working_dir_root_train_miccai_p3 import *
    elif import_mode == 'train_thoracic':
        from working_para.working_dir_root_train_Thoracic_p3 import *
    elif import_mode == 'eval_thoracic':
        # from working_para.working_dir_root_eval_Thoracic_p3 import *
        from working_para.working_dir_root_eval_Thoracic_p3 import *
 