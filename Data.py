import os
import sys
from random import shuffle, choice
from math import floor
from scipy import sparse

from time import time
import tensorflow as tf
import numpy as np
import smpl
import snug_utils as utils


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')


class Data:
    def __init__(self,  batch_size=10, mode='train'):
        """
        Args:
        - poses: path to .npy file with poses
        - shape: SMPL shape parameters for the subject
        - gender: 0 = female, 1 = male
        - batch_size: batch size
        - shuffle: shuffle
        """
        # Read sample list
        self._pose_path = "assets/CMU"
        self._poses, self._trans, self._trans_vel = self._get_pose()
        if self._poses.dtype == np.float64: self._poses = np.float32(self._poses)
        self._n_samples = self._poses.shape[0]
        # smpl
        self.SMPL = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

        self._shape = np.zeros(10, dtype=np.float32)
        # TF Dataset
        ds = tf.data.Dataset.from_tensor_slices((self._poses,self._trans,self._trans_vel))
        if mode == 'train': ds = ds.shuffle(self._n_samples)
        # ds = ds.map(self.tf_map, num_parallel_calls=batch_size)
        ds = ds.batch(batch_size=batch_size)
        self._iterator = ds

    def _get_pose(self):
        poses_array = np.zeros((1,72))
        trans_array = np.zeros((1,3))
        trans_vel_array = np.zeros((1,3))
        folder_num = 1
        file_num = 1
        k_folder = 0
        k_file = 0
        for folder in os.listdir(self._pose_path):
            k_folder+=1
            k_file = 0
            for file in os.listdir(os.path.join(self._pose_path,folder)):
                k_file+=1
                file_path = os.path.join(self._pose_path, folder, file)
                poses, trans, trans_vel = utils.load_motion(file_path)
                poses_array = np.concatenate((poses_array,poses),axis=0)
                trans_array = np.concatenate((trans_array, trans), axis=0)
                trans_vel_array = np.concatenate((trans_vel_array, trans_vel), axis=0)
                if k_file>=file_num:
                    break
            if k_folder>=folder_num:
                break
        poses_array = np.delete(poses_array,0,axis=0)
        trans_array = np.delete(trans_array,0,axis=0)
        trans_vel_array = np.delete(trans_vel_array,0,axis=0)
        remainder = poses_array.shape[0]%3
        poses_array = np.delete(poses_array, range(poses_array.shape[0]-remainder,poses_array.shape[0]),axis=0)
        poses_array = poses_array.reshape((-1,3,poses_array.shape[-1]))
        trans_array = np.delete(trans_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
        trans_array = trans_array.reshape((-1, 3, trans_array.shape[-1]))
        trans_vel_array = np.delete(trans_vel_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
        trans_vel_array = trans_vel_array.reshape((-1, 3, trans_vel_array.shape[-1]))

        return poses_array, trans_array, trans_vel_array



    def _next(self, pose):
        # compute body
        # while computing SMPL should be part of PBNS,
        # if it is in Data, it can be efficiently parallelized without overloading GPU
        G, B = self.SMPL.set_params(pose=pose.numpy(), beta=self._shape, with_body=True)

        return pose, G, B

    def tf_map(self, pose):
        return tf.py_function(func=self._next, inp=[pose], Tout=[tf.float32, tf.float32, tf.float32])