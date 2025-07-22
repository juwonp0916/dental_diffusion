import random
import numpy as np

class RandomRotate(object):  
    def __init__(self, angle=None, center=None, axis=["z"], p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.p = p 
        self.center = center


    def __call__(self, dentition_data):

        if random.random() < self.p:

            dentition_arr = dentition_data['d']
            bounds_cyl_centers = dentition_data['bcc']

            angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            if self.axis == "x":
                rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
            elif self.axis == "y":
                rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
            elif self.axis == "z":
                rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
            else:
                raise NotImplementedError

            if self.center is None:
                x_min, y_min, z_min = dentition_arr.min(axis=(0,1))
                x_max, y_max, z_max = dentition_arr.max(axis=(0,1))
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center

            dentition_arr = np.dot(dentition_arr - center, np.transpose(rot_t)) + center
            bounds_cyl_centers = np.dot(bounds_cyl_centers - center, np.transpose(rot_t)) + center

            z_max = np.max(dentition_arr[:,:,2],1)
            z_min = np.min(dentition_arr[:,:,2],1)
            new_h = z_max - z_min

            dentition_data['d'] = dentition_arr      
            dentition_data['bcc'] = bounds_cyl_centers
            dentition_data['bcs'][:,0] =  new_h

        return dentition_data



class RandomScale(object): 
    def __init__(self, scale=None, anisotropic=False, p=0.5):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic
        self.p = p

    def __call__(self, dentition_data):

        if random.random() < self.p:

            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )

            dentition_data['d'] = dentition_data['d'] * scale
            dentition_data['bcc'] = dentition_data['bcc'] * scale
            dentition_data['bcs'] = dentition_data['bcs'] * scale


        return dentition_data


class ShufflePoint(object): 
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, dentition_data):
        
        if random.random() < self.p:

            dentition_arr = dentition_data['d']

            num_teeth, num_points, _ = dentition_arr.shape

            for i in range(num_teeth):
                shuffle_index = np.arange(num_points)
                np.random.shuffle(shuffle_index)
                dentition_arr[i] = dentition_arr[i][shuffle_index]

            dentition_data['d'] = dentition_arr

        return dentition_data


class RandomMirror(object):  
    def __init__(self, p=0.5, center=None):
        self.center=center
        self.p = p 
        self.new_fdi_order = [26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1]

    def __call__(self, dentition_data): 

        if random.random() < self.p:


            dentition_arr = dentition_data['d']
            bounds_cyl_centers = dentition_data['bcc']
            bounds_cyl_size = dentition_data['bcs']


            dentition_arr_ = dentition_arr.copy()
            bounds_cyl_centers_ = bounds_cyl_centers.copy()
            bounds_cyl_size_ = bounds_cyl_size.copy()

            if self.center is None:
                x_min, y_min, z_min = dentition_arr_.min(axis=(0,1))
                x_max, y_max, z_max = dentition_arr_.max(axis=(0,1))
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center

            dentition_arr_ = dentition_arr_ - center
            dentition_arr_[..., 0] = -dentition_arr_[..., 0]
            dentition_arr_ = dentition_arr_ + center

            bounds_cyl_centers_ = bounds_cyl_centers_ - center
            bounds_cyl_centers_[..., 0] = -bounds_cyl_centers_[..., 0]
            bounds_cyl_centers_ = bounds_cyl_centers_ + center


            dentition_arr_ = dentition_arr_[self.new_fdi_order]
            bounds_cyl_centers_ = bounds_cyl_centers_[self.new_fdi_order]
            bounds_cyl_size_ = bounds_cyl_size_[self.new_fdi_order]

            dentition_data['d'] = dentition_arr_
            dentition_data['bcc'] = bounds_cyl_centers_
            dentition_data['bcs'] = bounds_cyl_size_

        return dentition_data