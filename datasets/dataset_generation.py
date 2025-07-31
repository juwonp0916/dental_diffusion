import os
import numpy as np
import torch as th
import json
from glob import glob


# TODO we want to include wisdom teeth as well. BUT only as context teeth.
# FDIS=[17,47,16,46,15,45,14,44,13,43,12,42,11,41,21,31,22,32,23,33,24,34,25,35,26,36,27,37]
FDIS = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
        38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]


class ToothDataset(th.utils.data.Dataset):
    def __init__(self, 
                 path, 
                 mode = 'train', # either train or val
                 tooth_npoints = 1024,
                 aug_transforms = None):
        super().__init__()

        self.mode = mode
        self.aug_transforms = aug_transforms
        self.tooth_npoints = tooth_npoints

        self.data_path = os.path.join(path, 'dentition')

        self.preloaded_dentition = {}
        patient_id_list = np.loadtxt(os.path.join(self.data_path, f'{self.mode}_patients.txt'), str)

        print('Preloading data...')

        for idx, patient_id in enumerate(patient_id_list): 
            
            self.preloaded_dentition[idx] = {'patient_id':patient_id, 'data':{}}

            # self.preloaded_dentition[idx] = {'patient_id':patient_id, 'data':{}, 'bounds':{}}
            for vert_path in glob(os.path.join(self.data_path, patient_id, 'verts', '*')):
                teeth_name = os.path.basename(vert_path)
                fdi = int(teeth_name.split('_')[-1].replace('.npy','').replace('FDI',''))
                if fdi not in FDIS:
                    continue

                self.preloaded_dentition[idx]['data'][fdi] = np.load(vert_path)

                # Bounding obj stuff
                # bound_path = vert_path.replace('verts','bounding').replace('.npy','_bounding.json')
                    
                # bound_json = json.load(open(bound_path,'r'))['cylinder']
                # self.preloaded_dentition[idx]['bounds'][fdi] = {
                #     'center':np.array([bound_json['cx'],bound_json['cy'],bound_json['cz']]),
                #     'size':np.array([bound_json['h'],bound_json['r']]),
                # }
            

    def __len__(self):
        return len(self.preloaded_dentition)

    def normalize_dentition(self, dentition_arr, bounds_cyl_c_arr, bounds_cyl_s_arr):

        #TODO 
        # Find a standard shift and scale values
        # Shift - Place the dentition data to the origin
        # Scale - ensure the dentition points are in (-1,1) range. 
        # allows more stable training
    # # 1. Use very rarely-missing teeth(11, 21, 31, 41) to find the shift and scale
    #     anchor_teeth = [11, 21, 31, 41]
    #     anchor_idx = None
    #     for fdi in anchor_teeth:
    #         if fdi in FDIS:
    #             idx = FDIS.index(fdi)
    #             # Check if this tooth is present (not all zeros)
    #             if not np.allclose(bounds_cyl_c_arr[idx], 0):
    #                 anchor_idx = idx
    #                 break

    #     if anchor_idx is not None:
    #         shift = bounds_cyl_c_arr[anchor_idx]
    #     else:
    #         # Fallback: use bounding box center
    #         bbox_min = np.min(bounds_cyl_c_arr, axis=0)
    #         bbox_max = np.max(bounds_cyl_c_arr, axis=0)
    #         shift = (bbox_min + bbox_max) / 2

    #     # Scale: use bounding box diagonal
    #     bbox_min = np.min(bounds_cyl_c_arr, axis=0)
    #     bbox_max = np.max(bounds_cyl_c_arr, axis=0)
    #     bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    #     scale = bbox_diag if bbox_diag > 1e-7 else 1.0

    #     dentition_arr_norm = (dentition_arr - shift) / scale
    #     bounds_cyl_c_arr_norm = (bounds_cyl_c_arr - shift) / scale
    #     bounds_cyl_s_arr_norm = bounds_cyl_s_arr / scale


    # 2. Use bounding box of the dentition points to find the scale
        bbox_min = np.min(bounds_cyl_c_arr, axis = 0)
        bbox_max = np.max(bounds_cyl_c_arr, axis = 0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)

        dentition_arr_shifted = dentition_arr - bbox_center
        bounds_cyl_c_arr_shifted = bounds_cyl_c_arr - bbox_center
        shift = bbox_center
        scale = bbox_diag
        # Avoid division by zero
        
        scale = bbox_diag if bbox_diag> 1e-7 else 1.0

        # Original values
        # shift = np.array([[-0.03630271,20.68396041,0.36647236]]) # find new value
        # scale = np.array([[10.5510643]]) # find new value

        dentition_arr_norm = (dentition_arr - shift) / scale
        bounds_cyl_c_arr_norm = (bounds_cyl_c_arr - shift) / scale
        bounds_cyl_s_arr_norm = (bounds_cyl_s_arr) / scale

        return dentition_arr_norm, bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm



    def sample_patient(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)

        dentition_list = []
        for idx in indexes:
            dentition_list.append(self.__getitem__(idx))
        return dentition_list

    def __getitem__(self, idx):
        dentition_data_dict = self.preloaded_dentition[idx]
        patient_id = dentition_data_dict['patient_id']

        try:


            dentition_arr = []
            # bounds_cyl_c_arr = []
            # bounds_cyl_s_arr = []

            # for fdi in FDIS:
            #     vert = dentition_data_dict['data'][fdi]
            #     random_index = np.random.randint(0, vert.shape[0], self.tooth_npoints) # uniform sampling of 1024 points per tooth
            #     dentition_arr.append(vert[random_index])

            for fdi in FDIS:
                if fdi in dentition_data_dict['data']:
                    vert = dentition_data_dict['data'][fdi]
                    random_index = np.random.randint(0, vert.shape[0], self.tooth_npoints)
                    dentition_arr.append(vert[random_index])
                else:
                    # Fill with zeros if tooth is missing
                    dentition_arr.append(np.zeros((self.tooth_npoints, 3), dtype=np.float32))

                # bounds_cyl_c_arr.append(dentition_data_dict['bounds'][fdi]['center'])
                # bounds_cyl_s_arr.append(dentition_data_dict['bounds'][fdi]['size'])


            dentition_arr = np.array(dentition_arr).reshape(len(FDIS), self.tooth_npoints, 3)
            # Newly created arrays
            dentition_arr_norm = (dentition_arr - dentition_arr.mean()) / dentition_arr.std()
            # bounds_cyl_c_arr = np.array(bounds_cyl_c_arr)
            # bounds_cyl_s_arr = np.array(bounds_cyl_s_arr)

            # assert bounds_cyl_c_arr.shape == (len(FDIS), 3)
            # assert bounds_cyl_s_arr.shape == (len(FDIS), 2)

            # dentition_arr_norm, bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm = self.normalize_dentition(dentition_arr,
            #                                                                                             bounds_cyl_c_arr,
            #                                                                                             bounds_cyl_s_arr)
                                                                                                                
        
            # if self.aug_transforms and self.mode == 'train':

            #     aug_out = self.aug_transforms({'d':dentition_arr_norm, 'bcc':bounds_cyl_c_arr_norm, 'bcs':bounds_cyl_s_arr_norm})
                
            #     dentition_arr_norm = aug_out['d']
            #     bounds_cyl_c_arr_norm = aug_out['bcc']
            #     bounds_cyl_s_arr_norm = aug_out['bcs']


            out = {
                'patient_id': dentition_data_dict['patient_id'], 
                'dentition_points': th.from_numpy(dentition_arr_norm).transpose(1,2).float(),
                # Auxiliary condition
                # 'bounds_cyl':th.from_numpy(np.concatenate([bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm], 1)).float()
            }
        except:
            print('PATIENT_ID', patient_id)
            raise Exception

        return out