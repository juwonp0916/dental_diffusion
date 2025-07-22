import os
import numpy as np
import torch as th
import json
from glob import glob


# TODO we want to include wisdom teeth as well. BUT only as context teeth.
FDIS=[17,47,16,46,15,45,14,44,13,43,12,42,11,41,21,31,22,32,23,33,24,34,25,35,26,36,27,37]


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
            self.preloaded_dentition[idx] = {'patient_id':patient_id, 'data':{}, 'bounds':{}}
            for vert_path in glob(os.path.join(self.data_path, patient_id, 'verts', '*')):
                teeth_name = os.path.basename(vert_path)
                fdi = int(teeth_name.split('_')[-1].replace('.npy','').replace('FDI',''))
                if fdi not in FDIS:
                    continue

                self.preloaded_dentition[idx]['data'][fdi] = np.load(vert_path)

                bound_path = vert_path.replace('verts','bounding').replace('.npy','_bounding.json')
                    
                bound_json = json.load(open(bound_path,'r'))['cylinder']
                self.preloaded_dentition[idx]['bounds'][fdi] = {
                    'center':np.array([bound_json['cx'],bound_json['cy'],bound_json['cz']]),
                    'size':np.array([bound_json['h'],bound_json['r']]),
                }
            

    def __len__(self):
        return len(self.preloaded_dentition)

    def normalize_dentition(self, dentition_arr, bounds_cyl_c_arr, bounds_cyl_s_arr):

        #TODO 
        # Find a standard shift and scale values
        # Shift - Place the dentition data to the origin
        # Scale - ensure the dentition points are in (-1,1) range. 
        # allows more stable training

        shift = np.array([[-0.03630271,20.68396041,0.36647236]]) # find new value
        scale = np.array([[10.5510643]]) # find new value

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
            bounds_cyl_c_arr = []
            bounds_cyl_s_arr = []

            for fdi in FDIS:
                vert = dentition_data_dict['data'][fdi]

                random_index = np.random.randint(0, vert.shape[0], self.tooth_npoints) # uniform sampling of 1024 points per tooth

                dentition_arr.append(vert[random_index])

                bounds_cyl_c_arr.append(dentition_data_dict['bounds'][fdi]['center'])
                bounds_cyl_s_arr.append(dentition_data_dict['bounds'][fdi]['size'])


            dentition_arr = np.array(dentition_arr).reshape(len(FDIS), self.tooth_npoints, 3)
            bounds_cyl_c_arr = np.array(bounds_cyl_c_arr)
            bounds_cyl_s_arr = np.array(bounds_cyl_s_arr)

            assert bounds_cyl_c_arr.shape == (len(FDIS), 3)
            assert bounds_cyl_s_arr.shape == (len(FDIS), 2)

            dentition_arr_norm, bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm = self.normalize_dentition(dentition_arr,
                                                                                                        bounds_cyl_c_arr,
                                                                                                        bounds_cyl_s_arr)
                                                                                                                
        
            if self.aug_transforms and self.mode == 'train':

                aug_out = self.aug_transforms({'d':dentition_arr_norm, 'bcc':bounds_cyl_c_arr_norm, 'bcs':bounds_cyl_s_arr_norm})
                
                dentition_arr_norm = aug_out['d']
                bounds_cyl_c_arr_norm = aug_out['bcc']
                bounds_cyl_s_arr_norm = aug_out['bcs']


            out = {
                'patient_id': dentition_data_dict['patient_id'], 
                'dentition_points': th.from_numpy(dentition_arr_norm).transpose(1,2).float(),
                'bounds_cyl':th.from_numpy(np.concatenate([bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm], 1)).float()
            }
        except:
            print('PATIENT_ID', patient_id)
            raise Exception

        return out