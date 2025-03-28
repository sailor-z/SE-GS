#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos
from utils.pose_utils import generate_random_poses_llff, generate_random_poses_360
from scene.cameras import PseudoCamera

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path
        print(f"args.source_path  is {args.source_path }")
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.bounds = None

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.source_path.find('llff') != -1:
                print("############ load llff ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            elif args.source_path.find('mipnerf360') != -1:
                print("############ load mipnerf360 ############")
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
            elif args.source_path.find('DTU') != -1:
                print("############ load DTU ############")
                scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, args.images, args.eval, args.n_views, rand_pcd=args.rand_pcd)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views, rand_pcd=args.rand_pcd)
        elif args.dataset == 'mvimgnet':
            scene_info = sceneLoadTypeCallbacks["MVImgNet"](args.source_path, args.images, args.eval, args.num_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", resolution_scale)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras", resolution_scale)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            pseudo_cams = []
            if args.source_path.find('llff') != -1:
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.source_path.find('mipnerf360') != -1:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            elif args.source_path.find('synthetic') != -1:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale])
            elif args.source_path.find('DTU') != -1:
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            elif args.dataset == 'mvimgnet':
                pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])


            view = self.train_cameras[resolution_scale][0]
            self.bounds = view.bounds
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
            self.pseudo_cameras[resolution_scale] = pseudo_cams


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.init_point_cloud = scene_info.point_cloud

    def save(self, iteration, path=None):
        if path is None:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        else:
            point_cloud_path = path
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]

class RenderScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, spiral=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}

        if 'scan' in args.source_path:
            scene_info = sceneLoadTypeCallbacks["SpiralDTU"](args.source_path)
        elif 'mvimgnet' in args.source_path:
            scene_info = sceneLoadTypeCallbacks["SpiralMVI"](args.source_path, args.num_views)
        else:
            scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Render Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = renderCameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            pass


    def getRenderCameras(self, scale=1.0):
        return self.test_cameras[scale]
