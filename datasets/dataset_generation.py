import os
import numpy as np
import torch as th
import json
from glob import glob


# TODO we want to include wisdom teeth as well. BUT only as context teeth.
FDIS=[17,47,16,46,15,45,14,44,13,43,12,42,11,41,21,31,22,32,23,33,24,34,25,35,26,36,27,37]

# FDI to embedding index mapping
# Upper right (11-17) -> indices 0-6
# Upper left (21-27) -> indices 7-13
# Lower left (31-37) -> indices 14-20
# Lower right (41-47) -> indices 21-27
# Missing teeth use original_missing_mask for identification
def create_fdi_to_idx_mapping():
    """
    Create mapping from FDI tooth numbers to embedding indices [0-27].
    FDI numbering: 11-17 (upper right), 21-27 (upper left), 31-37 (lower left), 41-47 (lower right)
    Mapping: 11-17 -> 0-6, 21-27 -> 7-13, 31-37 -> 14-20, 41-47 -> 21-27
    Note: We use ALL indices [0-27] for real teeth. Missing teeth are identified by the original_missing_mask.
    """
    fdi_to_idx = {}

    # Upper right quadrant: 11-17 -> 0-6
    for i in range(7):
        fdi_to_idx[11 + i] = i

    # Upper left quadrant: 21-27 -> 7-13
    for i in range(7):
        fdi_to_idx[21 + i] = 7 + i

    # Lower left quadrant: 31-37 -> 14-20
    for i in range(7):
        fdi_to_idx[31 + i] = 14 + i

    # Lower right quadrant: 41-47 -> 21-27
    for i in range(7):
        fdi_to_idx[41 + i] = 21 + i

    return fdi_to_idx

def create_idx_to_fdi_mapping():
    """Create reverse mapping from embedding indices to FDI numbers."""
    idx_to_fdi = {}

    # Upper right quadrant: 0-6 -> 11-17
    for i in range(7):
        idx_to_fdi[i] = 11 + i

    # Upper left quadrant: 7-13 -> 21-27
    for i in range(7):
        idx_to_fdi[7 + i] = 21 + i

    # Lower left quadrant: 14-20 -> 31-37
    for i in range(7):
        idx_to_fdi[14 + i] = 31 + i

    # Lower right quadrant: 21-27 -> 41-47
    for i in range(7):
        idx_to_fdi[21 + i] = 41 + i

    return idx_to_fdi

# Global mappings
FDI_TO_IDX = create_fdi_to_idx_mapping()
IDX_TO_FDI = create_idx_to_fdi_mapping()

def fdi_to_embedding_idx(fdi_number):
    """
    Convert FDI tooth number to embedding index [0-27].

    Key Design: All teeth (existing and missing) use their POSITIONAL embedding.
    - Embedding tells the model WHERE a tooth is anatomically (e.g., FDI 17 → index 6)
    - original_missing_mask tells the model IF the tooth exists (True = missing, False = present)
    - tooth_exists_indicator channel in the model input also signals existence (0 = missing, 1 = present)

    Example:
        - FDI 17 is missing → Still gets embedding index 6 (its anatomical position)
        - original_missing_mask[i] = True → Signals to model "this tooth doesn't exist"
        - Model sees: position=6 (where it should be), exists=False (but it's not there)

    This approach ensures:
        1. No ambiguity - each position has a unique embedding index
        2. Model knows spatial context even for missing teeth
        3. Mask system (not embedding) handles existence

    Args:
        fdi_number: FDI tooth number from FDIS list (11-17, 21-27, 31-37, 41-47)

    Returns:
        Embedding index [0-27] for the tooth's anatomical position
    """
    # Direct lookup - in normal operation, fdi_number is always from FDIS list
    # which contains only valid FDI numbers, so this will always succeed
    return FDI_TO_IDX.get(fdi_number, 0)

def embedding_idx_to_fdi(idx):
    """Convert embedding index [0-27] to FDI tooth number."""
    return IDX_TO_FDI.get(idx, 0)


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

        self.data_path = path

        self.preloaded_dentition = {}

        # Get all patient folders (both U and L variants)
        patient_folders = []
        for folder in os.listdir(self.data_path):
            if os.path.isdir(os.path.join(self.data_path, folder)) and ('U' in folder or 'L' in folder):
                patient_folders.append(folder)

        # Group by patient base ID (e.g., DBT1_0002L and DBT1_0002U -> DBT1_0002)
        patient_groups = {}
        for folder in patient_folders:
            base_id = folder[:-1]  # Remove 'U' or 'L' suffix
            if base_id not in patient_groups:
                patient_groups[base_id] = []
            patient_groups[base_id].append(folder)

        # Create patient list based on mode (using 80/20 split for now)
        all_patients = sorted(patient_groups.keys())
        split_idx = int(0.8 * len(all_patients))

        if self.mode == 'train':
            patient_id_list = all_patients[:split_idx]
        else:  # val mode
            patient_id_list = all_patients[split_idx:]

        print(f'Preloading {len(patient_id_list)} patients for {self.mode} mode...')

        for idx, patient_id in enumerate(patient_id_list):
            self.preloaded_dentition[idx] = {'patient_id':patient_id, 'data':{}}
            # self.preloaded_dentition[idx] = {'patient_id':patient_id, 'data':{}, 'bounds':{}}

            # Load from both upper and lower jaw folders for this patient
            for jaw_folder in patient_groups[patient_id]:
                jaw_path = os.path.join(self.data_path, jaw_folder, 'verts')
                if not os.path.exists(jaw_path):
                    continue

                for vert_path in glob(os.path.join(jaw_path, '*')):
                    teeth_name = os.path.basename(vert_path)
                    fdi = int(teeth_name.split('_')[-1].replace('.npy','').replace('FDI',''))
                    if fdi not in FDIS:
                        continue

                    self.preloaded_dentition[idx]['data'][fdi] = np.load(vert_path)

                # Bounding cylinder data - commented out for vertex-only processing
                # bound_path = vert_path.replace('verts','bounding').replace('.npy','_bounding.json')
                #
                # bound_json = json.load(open(bound_path,'r'))['cylinder']
                # self.preloaded_dentition[idx]['bounds'][fdi] = {
                #     'center':np.array([bound_json['cx'],bound_json['cy'],bound_json['cz']]),
                #     'size':np.array([bound_json['h'],bound_json['r']]),
                # }
            

    def __len__(self):
        return len(self.preloaded_dentition)

    def normalize_dentition(self, dentition_arr):
        """
        Normalize dentition using fixed shift/scale from ICP-registered manual templates.

        Origin is at the middle of full dentition (not between central incisors).
        Based on manual templates:
        - Upper jaw: DBT1_0300U (center: [-1.19, 26.69, -11.82])
        - Lower jaw: DBT1_0655L (center: [-0.24, 24.01, -1.45])
        - Combined middle: [-0.72, 25.35, -6.64]
        - Max range: ~78.4 mm (X-axis)

        Normalization to [-1, +1] range based on max extent of manual templates.

        Args:
            dentition_arr: (28, N, 3) array of point clouds for all teeth

        Returns:
            dentition_arr_norm: (28, N, 3) normalized point clouds
            existing_teeth_mask: (28,) boolean mask indicating which teeth exist
        """
        # Fixed shift: middle point of full 28-teeth dentition from ICP-registered templates
        shift = np.array([[-0.72, 25.35, -6.64]])

        # Fixed scale: half of max range for [-1, +1] normalization
        # Using 39.2 = 78.4 / 2 (where 78.4 is max X-range from templates)
        scale = 39.2

        # Apply normalization: (data - shift) / scale
        dentition_arr_norm = (dentition_arr - shift) / scale

        # Create existing teeth mask: detect teeth with real data vs noise
        # Noise-filled missing teeth have very small ORIGINAL values (~0.05 scale)
        # Real teeth have much larger coordinate values (>5mm in any dimension)
        # Check BEFORE normalization to avoid shift effects
        existing_teeth_mask = np.any(np.abs(dentition_arr) > 1.0, axis=(1, 2))

        return dentition_arr_norm, existing_teeth_mask

    def _validate_normalization(self, original_dentition, normalized_dentition, existing_teeth_mask):
        """
        Validate that normalization preserves spatial relationships.
        """
        if np.sum(existing_teeth_mask) < 2:
            return  # Cannot validate spatial relationships with less than 2 teeth

        existing_indices = np.where(existing_teeth_mask)[0]

        # Check that relative distances between teeth are preserved proportionally
        for i in range(len(existing_indices)):
            for j in range(i + 1, len(existing_indices)):
                idx1, idx2 = existing_indices[i], existing_indices[j]

                # Calculate centers of teeth
                orig_center1 = np.mean(original_dentition[idx1], axis=0)
                orig_center2 = np.mean(original_dentition[idx2], axis=0)
                norm_center1 = np.mean(normalized_dentition[idx1], axis=0)
                norm_center2 = np.mean(normalized_dentition[idx2], axis=0)

                # Calculate relative distances
                orig_dist = np.linalg.norm(orig_center1 - orig_center2)
                norm_dist = np.linalg.norm(norm_center1 - norm_center2)

                # Check that distance ratio is consistent (allowing for some scaling)
                if orig_dist > 1e-6 and norm_dist > 1e-6:
                    ratio = norm_dist / orig_dist
                    # Allow reasonable scaling but ensure relative positioning is preserved
                    if ratio < 0.1 or ratio > 10.0:
                        print(f"Warning: Large distance ratio {ratio:.3f} between teeth {idx1} and {idx2}")

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
            fdi_list = []  # NEW: Track actual FDI numbers
            # bounds_cyl_c_arr = []
            # bounds_cyl_s_arr = []
            original_missing_mask = []

            for fdi in FDIS:
                if fdi in dentition_data_dict['data']:
                    # CASE 1: Tooth exists in dataset
                    vert = dentition_data_dict['data'][fdi]
                    random_index = np.random.randint(0, vert.shape[0], self.tooth_npoints)
                    dentition_arr.append(vert[random_index])
                    fdi_list.append(fdi_to_embedding_idx(fdi))  # Positional embedding [0-27]
                    # bounds_cyl_c_arr.append(dentition_data_dict['bounds'][fdi]['center'])
                    # bounds_cyl_s_arr.append(dentition_data_dict['bounds'][fdi]['size'])
                    original_missing_mask.append(False)  # Signals: tooth is PRESENT
                else:
                    # CASE 2: Tooth is naturally missing
                    # Fill with small noise (will be replaced with even smaller noise after normalization)
                    noise = np.random.normal(loc=0.0, scale=0.05, size=(self.tooth_npoints, 3))
                    dentition_arr.append(noise)

                    # IMPORTANT: Missing teeth ALSO get their positional embedding index
                    # This tells the model WHERE the tooth should be anatomically
                    # The mask (below) tells the model that it doesn't actually exist
                    fdi_list.append(fdi_to_embedding_idx(fdi))  # Same positional embedding [0-27]

                    # Use zero bounds for missing teeth
                    # bounds_cyl_c_arr.append(np.zeros(3))
                    # bounds_cyl_s_arr.append(np.zeros(2))

                    # The mask signals to the model: "this tooth doesn't exist"
                    original_missing_mask.append(True)  # Signals: tooth is MISSING


            dentition_arr = np.array(dentition_arr).reshape(len(FDIS), self.tooth_npoints, 3)
            # bounds_cyl_c_arr = np.array(bounds_cyl_c_arr)
            # bounds_cyl_s_arr = np.array(bounds_cyl_s_arr)

            # assert bounds_cyl_c_arr.shape == (len(FDIS), 3)
            # assert bounds_cyl_s_arr.shape == (len(FDIS), 2)

            dentition_arr_norm, existing_teeth_mask = self.normalize_dentition(dentition_arr)

            # CRITICAL FIX: Fill naturally missing teeth with small noise AFTER normalization
            # This ensures missing teeth have near-zero values in the normalized space
            for tooth_idx in range(len(FDIS)):
                if original_missing_mask[tooth_idx]:
                    # Replace with small noise in normalized space
                    dentition_arr_norm[tooth_idx] = np.random.normal(loc=0.0, scale=0.01, size=(self.tooth_npoints, 3))
            # dentition_arr_norm, bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm = self.normalize_dentition(dentition_arr,
            #                                                                                             bounds_cyl_c_arr,
            #                                                                                             bounds_cyl_s_arr)
                                                                                                                
        
            # Data augmentation - commented out for vertex-only processing
            # if self.aug_transforms and self.mode == 'train':
            #     aug_out = self.aug_transforms({'d':dentition_arr_norm, 'bcc':bounds_cyl_c_arr_norm, 'bcs':bounds_cyl_s_arr_norm})
            #     dentition_arr_norm = aug_out['d']
            #     bounds_cyl_c_arr_norm = aug_out['bcc']
            #     bounds_cyl_s_arr_norm = aug_out['bcs']


            out = {
                'patient_id': dentition_data_dict['patient_id'],
                'dentition_points': th.from_numpy(dentition_arr_norm).transpose(1,2).float(),
                'fdi_indices': th.tensor(fdi_list, dtype=th.long),  # (28,) embedding indices [0-27]
                # 'bounds_cyl': th.from_numpy(np.concatenate([bounds_cyl_c_arr_norm, bounds_cyl_s_arr_norm], 1)).float(),
                'original_missing_mask': th.from_numpy(np.array(original_missing_mask)).bool()
            }
        except:
            print('PATIENT_ID', patient_id)
            raise Exception

        return out