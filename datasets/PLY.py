#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling PLY dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Leith Webster
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
from time import time as time_time
# import numpy as np
from numpy import (uint8 as np_uint8, int64 as np_int64, array as np_array, float64 as np_float64, zeros as np_zeros,
                    square as np_square, float32 as np_float32, hstack as np_hstack, concatenate as np_concatenate, 
                    int32 as np_int32, stack as np_stack, ones_like as np_ones_like, vstack as np_vstack, 
                    squeeze as np_squeeze, ceil as np_ceil, pi as np_pi, sum as np_sum, bincount as np_bincount,
                    abs as np_abs, max as np_max, cumsum as np_cumsum)
from numpy.random import rand as np_random_rand, normal as np_random_normal
from pickle import dump as pickle_dump
# import torch
from multiprocessing import Lock


# OS functions
from os import makedirs
from os.path import exists, join, dirname, splitext, split

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler #, get_worker_info
from torch import (tensor as torch_tensor, float32 as torch_float32, from_numpy as torch_from_numpy,
                   argmin as torch_argmin, int32 as torch_int32)
# from utils.mayavi_visu import *
from sklearn.neighbors import KDTree
from utils.ply import read_ply

from datasets.common import grid_subsampling
from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class PLYDataset(PointCloudDataset):
    """Class to handle PLY dataset."""

    def __init__(self, config, infile, set='training', use_potentials=True, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'PLY')

        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'chair',
                               8: 'table',
                               9: 'bookcase',
                               10: 'sofa',
                               11: 'board',
                               12: 'clutter'}
        
        self.label_to_colour = {0: [np_uint8(0),np_uint8(255),np_uint8(0)],
                                1: [np_uint8(0),np_uint8(0),np_uint8(255)],
                                2: [np_uint8(0),np_uint8(255),np_uint8(255)],
                                3: [np_uint8(255),np_uint8(255),np_uint8(0)],
                                4: [np_uint8(255),np_uint8(0),np_uint8(255)],
                                5: [np_uint8(100),np_uint8(100),np_uint8(255)],
                                6: [np_uint8(200),np_uint8(200),np_uint8(100)],
                                7: [np_uint8(170),np_uint8(120),np_uint8(200)],
                                8: [np_uint8(255),np_uint8(0),np_uint8(0)],
                                9: [np_uint8(200),np_uint8(100),np_uint8(100)],
                                10: [np_uint8(10),np_uint8(200),np_uint8(100)],
                                11: [np_uint8(200),np_uint8(200),np_uint8(200)],
                                12: [np_uint8(50),np_uint8(50),np_uint8(50)]}

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of files
        self.file = infile

        # Dataset folder
        self.path = dirname(infile) + '/'

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Proportion of validation scenes
        # self.cloud_name = '2025-10-27_Livox_Viewer_Frame_Test'
        self.cloud_name = split(splitext(infile)[0])[1]

        # Number of models used per epoch
        if self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for PLY data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return

        ################
        # Load ply files
        ################

        # List of files
        # self.file = join(self.path, self.cloud_name + '.ply')

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_label = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch_tensor([1], dtype=torch_float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch_from_numpy(np_random_rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch_argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch_from_numpy(np_array(self.argmin_potentials, dtype=np_int64))
            self.min_potentials = torch_from_numpy(np_array(self.min_potentials, dtype=np_float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch_tensor([0 for _ in range(config.input_threads)], dtype=torch_int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            N = config.epoch_steps * config.batch_num
            self.epoch_inds = torch_from_numpy(np_zeros((2, N), dtype=np_int64))
            self.epoch_i = torch_from_numpy(np_zeros((1,), dtype=np_int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        return

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        return self.potential_item()

    def potential_item(self):

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                point_ind = int(self.argmin_potentials[0])

                # Get potential points from tree structure
                pot_points = np_array(self.pot_trees[0].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                center_point += np_random_normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[0].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np_square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                tukeys = np_square(1 - d2s / np_square(self.config.in_radius))
                tukeys[d2s > np_square(self.config.in_radius)] = 0
                self.potentials[0][pot_inds] += tukeys
                min_ind = torch_argmin(self.potentials[0])
                self.min_potentials[[0]] = self.potentials[0][min_ind]
                self.argmin_potentials[[0]] = min_ind

            # Get points from tree structure
            points = np_array(self.input_trees[0].data, copy=False)


            # Indices of points in input region
            input_inds = self.input_trees[0].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np_float32)
            input_colors = self.input_colors[0][input_inds]
            # input_label = self.input_label[0][input_inds]
            # input_label = np_array([self.label_to_idx[l] for l in input_label])

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np_random_rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np_hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np_float32)

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += []
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [0]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

        ###################
        # Concatenate batch
        ###################

        stacked_points = np_concatenate(p_list, axis=0)
        features = np_concatenate(f_list, axis=0)
        # labels = np_concatenate(l_list, axis=0)
        labels = []
        point_inds = np_array(i_list, dtype=np_int32)
        cloud_inds = np_array(ci_list, dtype=np_int32)
        input_inds = np_concatenate(pi_list, axis=0)
        stack_lengths = np_array([pp.shape[0] for pp in p_list], dtype=np_int32)
        scales = np_array(s_list, dtype=np_float32)
        rots = np_stack(R_list, axis=0)

        # Input features
        stacked_features = np_ones_like(stacked_points[:, :1], dtype=np_float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np_hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np_hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        return input_list

    def load_subsampled_clouds(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        # Restart timer
        t0 = time_time()

        # Name of the input files
        KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(self.cloud_name))

        print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(self.cloud_name, dl))

        # Read ply file
        data = read_ply(self.file)
        # points = np_vstack((data['x'], data['y'], data['z'])).T
        points = np_vstack((-data['x'], data['z'], data['y'])).T
        colors = np_vstack((data['red'], data['green'], data['blue'])).T
        # labels = data['class']

        # Subsample cloud
        sub_points, sub_colors = grid_subsampling(points,
                                                                features=colors,
                                                                labels=None,
                                                                sampleDl=dl)

        # Rescale float color and squeeze label
        sub_colors = sub_colors / 255
        # sub_labels = np_squeeze(sub_labels)

        # Get chosen neighborhoods
        search_tree = KDTree(sub_points, leaf_size=10)

        # Save KDTree
        with open(KDTree_file, 'wb') as f:
            pickle_dump(search_tree, f)

        # Fill data containers
        self.input_trees += [search_tree]
        self.input_colors += [sub_colors]
        # self.input_label += [sub_labels]

        size = sub_colors.shape[0] * 4 * 7
        print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time_time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time_time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            # Get cloud name
            cloud_name = self.cloud_name

            # Name of the input files
            coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

            # Subsample cloud
            sub_points = np_array(self.input_trees[cloud_ind].data, copy=False)
            coarse_points = grid_subsampling(sub_points.astype(np_float32), sampleDl=pot_dl)

            # Get chosen neighborhoods
            search_tree = KDTree(coarse_points, leaf_size=10)

            # Save KDTree
            with open(coarse_KDTree_file, 'wb') as f:
                pickle_dump(search_tree, f)

            # Fill data containers
            self.pot_trees += [search_tree]
            cloud_ind += 1

        print('Done in {:.1f}s'.format(time_time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices

            # Restart timer
            t0 = time_time()

            # File name for saving
            proj_file = join(tree_path, '{:s}_proj.pkl'.format(self.cloud_name))

            data = read_ply(self.file)

            # points = np_vstack((data['x'], data['y'], data['z'])).T
            points = np_vstack((-data['x'], data['z'], data['y'])).T
            # labels = data['class']

            # Compute projection inds
            idxs = self.input_trees[0].query(points, return_distance=False)
            proj_inds = np_squeeze(idxs).astype(np_int32)

            # Save
            with open(proj_file, 'wb') as f:
                # pickle_dump([proj_inds, labels], f)
                pickle_dump([proj_inds], f)

            self.test_proj += [proj_inds]
            # self.validation_labels += [labels]
            print('{:s} done in {:.1f}s'.format(self.cloud_name, time_time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        # return np_vstack((data['x'], data['y'], data['z'])).T
        return np_vstack((-data['x'], data['z'], data['y'])).T
    
    def load_evaluation_points_and_colours(self, file_path):
        """
        Load points and colours (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        # return np_vstack((data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'])).T

        return np_vstack((-data['x'], data['z'], data['y'], data['red'], data['green'], data['blue'])).T


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class PLYSampler(Sampler):
    """Sampler for PLY"""

    def __init__(self, dataset: PLYDataset):
        Sampler.__init__(self)
        # Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # Generator loop
        for i in range(self.N):
            yield i

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time_time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np_ceil(4 / 3 * np_pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np_zeros((self.dataset.config.num_layers, hist_n), dtype=np_int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time_time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np_sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np_bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np_vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np_abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np_max(np_abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time_time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                                estim_b,
                                                int(self.dataset.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np_cumsum(neighb_hists.T, axis=0)
            percentiles = np_sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np_sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle_dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle_dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time_time() - t0))
        return


class PLYCustomBatch:
    """Custom batch definition with memory pinning for PLY"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch_from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch_from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch_from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch_from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch_from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch_from_numpy(input_list[ind])
        ind += 1
        # self.labels = torch_from_numpy(input_list[ind])
        ind += 1
        self.scales = torch_from_numpy(input_list[ind])
        ind += 1
        self.rots = torch_from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch_from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch_from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch_from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        # self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        # self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

def PLYCollate(batch_data):
    return PLYCustomBatch(batch_data)
