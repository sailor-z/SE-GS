#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import cv2
import os
import json
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from pytorch3d.ops import knn_points
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, matrix_to_rotation_6d
import random
from random import randint
import open3d as o3d
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, loss_photometric
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.cameras import PseudoCamera
from utils.graphics_utils import get_intrinsic
from utils.general_utils import safe_state
from utils.pose_utils import generate_pseudo_views, slerp
from utils.general_utils import get_expon_noise_ration_func

import uuid
import wandb
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.hub.set_dir("/scratch/cvlab/home/chzhao/ModelZoo/pretrained_models")

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def get_cam_dis(view, views):
    T = torch.from_numpy(view.T).cuda().float()
    R = torch.from_numpy(view.R.transpose()).cuda().float()
    Ts = torch.from_numpy(np.asarray([view.T for view in views])).cuda().float()
    Rs = np.asarray([view.R.transpose() for view in views])
    Rs = torch.from_numpy(Rs).cuda().float()

    sim_R = (torch.sum(R.reshape(1, 9) * Rs.reshape(-1, 9), dim=1).clamp(-1, 3) - 1) / 2
    R_dis = 180 * torch.arccos(sim_R) / np.pi

    T_dis = torch.norm(T[None] - Ts, p=2, dim=-1)

    return R_dis, T_dis

@torch.no_grad()
def find_reset_gaussians_ratio(args, gaussians, view, radii, val_std, rendered_img, iteration, viz=False, ratio=0.1, thr=5e-2):
    pts3d = gaussians.get_xyz

    K = torch.from_numpy(get_intrinsic(view)).cuda().float()
    R = torch.from_numpy(view.R.transpose()).cuda().float()
    T = torch.from_numpy(view.T[:, None]).cuda().float()

    proj_pts3d = K @ (R @ pts3d.transpose(0, 1) + T)
    proj_pts3d = proj_pts3d.transpose(0, 1)
    depth = proj_pts3d[:, 2]
    proj_pts = proj_pts3d[:, :2] / depth[:, None].clamp(min=1e-6)

    valid_maskx = (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < view.image_width)
    valid_masky = (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < view.image_height)
    valid_mask = torch.logical_and(valid_maskx, valid_masky)

    val_std_sorted = torch.sort(val_std.flatten(0), descending=True)[0]
    data_thr = val_std_sorted[int(ratio * view.image_width * view.image_height)]
    thr = max(data_thr, thr)

    std_mask = val_std > thr

    if torch.all(std_mask == 0):
        return pts3d.new_zeros(pts3d.shape[0])

    pts2d = torch.stack(torch.meshgrid(torch.arange(view.image_height, device="cuda"), torch.arange(view.image_width, device="cuda")), -1).float()
    pts2d = pts2d[..., (1, 0)]
    pts2d = pts2d[std_mask]

    dist, _, _ = knn_points(proj_pts[None], pts2d[None], K=1)
    dist = dist.squeeze()

    reset_mask = dist < radii
    reset_mask = torch.logical_and(valid_mask, reset_mask)

    return reset_mask


def training_scene(dataset, opt, pipe, args):
    # implenmetation of more than 2 3d gaussian radiance fields currently is not supported in this code
    assert args.gaussiansN >= 1 and args.gaussiansN <=2
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    init_gaussians = GaussianModel(args)
    scene = Scene(dataset, init_gaussians, shuffle=False)
    init_gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    GsDict = {}
    GsDict["gs0"] = deepcopy(init_gaussians)
    GsDict["gs1"] = GaussianModel(args)
    GsDict["gs1"].create_from_pcd(scene.init_point_cloud, scene.cameras_extent)
    GsDict["gs1"].training_setup(opt)

    print(f"GsDict.keys() is {GsDict.keys()}")

    ### pseudo validation set
    val_stack0, val_stack1 = [], []

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    training_views = scene.getTrainCameras().copy()
    pseudo_views = generate_pseudo_views(training_views, num_views=10000)

    noise_until_iter = opt.iterations - args.noise_interval
    iter_delay = args.noise_from_iter + args.noise_interval * args.len_stack

    noise_ratio_sche = get_expon_noise_ration_func(args.noise_ratio_max, args.noise_ratio_min, max_steps=noise_until_iter-iter_delay)

    pseudo_val_views = []
    while len(pseudo_val_views) < args.num_val:
        idx = randint(0, len(pseudo_views)-1)
        pseudo_val_views.append(pseudo_views[idx])
        pseudo_views.pop(idx)

    pseudo_val_imgs = [[] for i in range(args.num_val)]

    viewpoint_stack = None
    pseudo_stack_co = None
    ema_loss_for_log = 0.0
    first_iter += 1

    output_path = os.path.join(args.model_path, str(args.num_views) + "_views")
    os.makedirs(output_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            GsDict["gs0"].oneupSHdegree()
            GsDict["gs1"].oneupSHdegree()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = training_views.copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        RenderDict = {}
        LossDict = {}
        logDict = {}

        # render for main viewpoint
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        for idx in [0, 1]:
            RenderDict[f"render_pkg_gs{idx}"] = render(viewpoint_cam, GsDict[f"gs{idx}"], pipe, bg)
            RenderDict[f"image_gs{idx}"] = RenderDict[f"render_pkg_gs{idx}"]["render"]
            RenderDict[f"depth_gs{idx}"] = RenderDict[f"render_pkg_gs{idx}"]["depth"]
            RenderDict[f"alpha_gs{idx}"] = RenderDict[f"render_pkg_gs{idx}"]["alpha"]
            RenderDict[f"viewspace_point_tensor_gs{idx}"] = RenderDict[f"render_pkg_gs{idx}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{idx}"] = RenderDict[f"render_pkg_gs{idx}"]["visibility_filter"]
            RenderDict[f"radii_gs{idx}"] = RenderDict[f"render_pkg_gs{idx}"]["radii"]

            LossDict[f"loss_gs{idx}"] = loss_photometric(RenderDict[f"image_gs{idx}"], gt_image, opt=opt)

        loss_gs_pseudo0, loss_gs_pseudo1 = None, None
        if not args.onlyrgb:
            if iteration % args.sample_pseudo_interval == 0 and iteration <= args.end_sample_pseudo:
                if not pseudo_stack_co:
                    pseudo_stack_co = pseudo_views.copy()
                pseudo_cam_co = pseudo_stack_co.pop(randint(0, len(pseudo_stack_co) - 1))

                for idx in [0, 1]:
                    RenderDict[f"render_pkg_pseudo_co_gs{idx}"] = render(pseudo_cam_co, GsDict[f"gs{idx}"], pipe, bg)
                    RenderDict[f"image_pseudo_co_gs{idx}"] = RenderDict[f"render_pkg_pseudo_co_gs{idx}"]["render"]
                    RenderDict[f"depth_pseudo_co_gs{idx}"] = RenderDict[f"render_pkg_pseudo_co_gs{idx}"]["depth"]

                if iteration >= args.start_sample_pseudo and args.coreg:
                    loss_gs_pseudo0 = loss_photometric(RenderDict["image_pseudo_co_gs0"], RenderDict["image_pseudo_co_gs1"].clone().detach(), opt=opt)
                    LossDict["loss_gs0"] += loss_gs_pseudo0
                    loss_gs_pseudo1 = loss_photometric(RenderDict["image_pseudo_co_gs1"], RenderDict["image_pseudo_co_gs0"].clone().detach(), opt=opt)
                    LossDict["loss_gs1"] += loss_gs_pseudo1

        loss = LossDict["loss_gs0"]
        for idx in [0, 1]:
            LossDict[f"loss_gs{idx}"].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "num_pts": GsDict["gs0"].get_xyz.shape[0], "num_views": len(viewpoint_stack)})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration > first_iter and (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                pcd_path = os.path.join(output_path, "point_cloud/iteration_{}".format(iteration))
                GsDict["gs0"].save_ply(os.path.join(pcd_path, "point_cloud.ply"))

                pcd_path = os.path.join(output_path, "point_cloud_gs2/iteration_{}".format(iteration))
                GsDict["gs1"].save_ply(os.path.join(pcd_path, "point_cloud.ply"))

            if iteration > first_iter and (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((GsDict["gs0"].capture(), iteration), output_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if iteration < opt.densify_until_iter:
                for idx in [0, 1]:
                    # Keep track of max radii in image-space for pruning
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{idx}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{idx}"]
                    radii = RenderDict[f"radii_gs{idx}"]
                    GsDict[f"gs{idx}"].max_radii2D[visibility_filter] = torch.max(GsDict[f"gs{idx}"].max_radii2D[visibility_filter], radii[visibility_filter])
                    GsDict[f"gs{idx}"].add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    for idx in [0, 1]:
                        # density and prune
                        GsDict[f"gs{idx}"].densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                for key in GsDict.keys():
                    GsDict[key].optimizer.step()
                    GsDict[key].optimizer.zero_grad(set_to_none = True)

            ### co-pruning
            if args.coprune and iteration > opt.densify_from_iter and iteration % 500 == 0:
                source_cloud = o3d.geometry.PointCloud()
                source_cloud.points = o3d.utility.Vector3dVector(GsDict["gs0"].get_xyz.clone().cpu().numpy())
                target_cloud = o3d.geometry.PointCloud()
                target_cloud.points = o3d.utility.Vector3dVector(GsDict["gs1"].get_xyz.clone().cpu().numpy())
                trans_matrix = np.identity(4)
                threshold = args.coprune_threshold

                eval = o3d.pipelines.registration.evaluate_registration(source_cloud, target_cloud, threshold, trans_matrix)
                correspondence = np.array(eval.correspondence_set)
                mask_consistent = torch.zeros((GsDict["gs0"].get_xyz.shape[0], 1)).cuda()
                mask_consistent[correspondence[:, 0], :] = 1
                mask_inconsistent = ~(mask_consistent.bool())
                GsDict["gs0"].prune_from_mask(mask_inconsistent.squeeze(), iter=iteration)

                eval = o3d.pipelines.registration.evaluate_registration(target_cloud, source_cloud, threshold, trans_matrix)
                correspondence = np.array(eval.correspondence_set)
                mask_consistent = torch.zeros((GsDict["gs1"].get_xyz.shape[0], 1)).cuda()
                mask_consistent[correspondence[:, 0], :] = 1
                mask_inconsistent = ~(mask_consistent.bool())
                GsDict["gs1"].prune_from_mask(mask_inconsistent.squeeze(), iter=iteration)

            ### perturbation
            if iteration > args.noise_from_iter and iteration < noise_until_iter and iteration % args.noise_interval == 0:
                pseudo_val_radii = []
                for idx, pseudo_val_view in enumerate(pseudo_val_views):
                    ValRenderDict = render(pseudo_val_view, GsDict["gs0"], pipe, bg)
                    pseudo_val_img = ValRenderDict["render"]
                    ValRenderDict = render(pseudo_val_view, GsDict["gs1"], pipe, bg)
                    pseudo_val_radii.append(ValRenderDict["radii"])

                    if len(pseudo_val_imgs[idx]) < args.len_stack:
                        pseudo_val_imgs[idx].append(pseudo_val_img)
                    else:
                        pseudo_val_imgs[idx].pop(0)
                        pseudo_val_imgs[idx].append(pseudo_val_img)

                if len(pseudo_val_imgs[0]) == args.len_stack:
                    val_stack = [torch.stack(pseudo_val_imgs[idx], dim=0) for idx in range(len(pseudo_val_imgs))]
                    val_stack = torch.stack(val_stack, dim=0)
                    val_std = val_stack.std(dim=1).mean(dim=1)

                    ### local patch averaging
                    filters = val_std.new_ones([1, 1, 5, 5])
                    val_std = F.conv2d(val_std[:, None], filters, padding=2).squeeze(1) / 25.0

                    reset_mask = [find_reset_gaussians_ratio(args, GsDict["gs1"], pseudo_val_views[idx], pseudo_val_radii[idx], \
                    val_std[idx], pseudo_val_imgs[idx][-1], iteration, viz=True, ratio=args.reset_ratio, thr=args.reset_thr) for idx in range(len(pseudo_val_imgs))]

                    reset_mask = torch.stack(reset_mask).float()
                    reset_mask = reset_mask.sum(dim=0) > 0

                    noise_ratio = noise_ratio_sche(iteration - iter_delay)
                    GsDict["gs1"].reset_attributes(reset_mask, ratio=noise_ratio)

                    print("%d of %d have been reset with a noise ratio of %.4f" % (reset_mask.sum(), reset_mask.shape[0], noise_ratio))

            for key in GsDict.keys():
                GsDict[key].update_learning_rate(iteration)
                if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                        iteration > args.start_sample_pseudo:
                    GsDict[key].reset_opacity()

def training(args, lp, op, pp):
    args.source_path = args.data_path
    fused_file = os.path.exists(os.path.join(args.data_path, f'{args.n_views}_views', 'dense', 'fused.ply'))
    bound_file = os.path.exists(os.path.join(args.data_path, f'{args.n_views}_views', 'dense', 'poses_bounds.npy'))
    if fused_file == False or  bound_file == False:
        print("Skip the sequence since COLMAP failed")

    training_scene(lp.extract(args), op.extract(args), pp.extract(args), args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--data_path',  type=str, default=None)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 5000, 7000, 10000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--checkpoint", type=str, default = None)
    parser.add_argument('--gaussiansN', type=int, default=1)
    parser.add_argument("--onlyrgb", action='store_true', default=False)
    parser.add_argument("--coreg", action='store_true', default=False)
    parser.add_argument("--coprune", action='store_true', default=False)
    parser.add_argument('--coprune_threshold', type=int, default=5)
    parser.add_argument('--len_stack', type=int, default=3)
    parser.add_argument('--reset_thr', type=float, default=1e-2)
    parser.add_argument('--reset_ratio', type=float, default=0.05)
    parser.add_argument('--num_val', type=int, default=24)
    parser.add_argument('--noise_from_iter', type=int, default=500)
    parser.add_argument('--noise_interval', type=int, default=500)
    parser.add_argument('--noise_ratio_max', type=float, default=0.02)
    parser.add_argument('--noise_ratio_min', type=float, default=0.01)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(args, lp, op, pp)

    # All done
    print("\nTraining complete.")
