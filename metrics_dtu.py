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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from skimage.metrics import structural_similarity

from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, avge
from argparse import ArgumentParser

def custom_sort(file_name):
    return int(file_name.split('_')[1])

def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    idx = 0
    render_list = os.listdir(renders_dir)
    render_list = sorted(render_list, key=custom_sort)
    for fname in render_list:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(mask_dir / '{0:05d}.png'.format(idx))
        mask = mask.resize(gt.size)
        mask = tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda()
        mask_bin = (mask == 1.)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda() * mask + (1-mask))
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda() * mask + (1-mask))
        masks.append(mask_bin)
        image_names.append(fname)
        idx += 1
    return renders, gts, image_names, masks




def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for test_dir in [test_dir]:
            dataset = test_dir.stem
            for method in os.listdir(test_dir):
                print("Method:", method, dataset)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                mask_dir = Path(scene_dir) / "mask"
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names, masks = readImages(renders_dir, gt_dir, mask_dir)

                os.makedirs(mask_dir / "masked", exist_ok=True)

                for idx, img in enumerate(tqdm(renders, desc="save", ascii=True, dynamic_ncols=True)):
                    torchvision.utils.save_image(img, os.path.join(mask_dir / "masked", '{0:05d}'.format(idx) + ".png"))

                ssims = []
                ssims_sk = []
                psnrs = []
                lpipss = []
                avges = []
                avges_sk = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    ssims_sk.append(structural_similarity(renders[idx][0].permute(1,2,0).cpu().numpy(), gts[idx][0].permute(1,2,0).cpu().numpy(), channel_axis=2, data_range=1.0))
                    psnrs.append(psnr(renders[idx][masks[idx]][None, ...], gts[idx][masks[idx]][None, ...]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    avges.append(avge(torch.tensor(ssims[idx]), torch.tensor(psnrs[idx]), torch.tensor(lpipss[idx])))
                    avges_sk.append(avge(torch.tensor(ssims_sk[idx]), torch.tensor(psnrs[idx]), torch.tensor(lpipss[idx])))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  SSIM_sk : {:>12.7f}".format(torch.tensor(ssims_sk).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  AVGE: {:>12.7f}".format(torch.tensor(avges).mean(), ".5"))
                print("  AVGE_sk: {:>12.7f}".format(torch.tensor(avges_sk).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "SSIM_sk": torch.tensor(ssims_sk).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "AVGE": torch.tensor(avges).mean().item(),
                                                        "AVGE_sk": torch.tensor(avges_sk).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "SSIM_sk": {name: ssim for ssim, name in zip(torch.tensor(ssims_sk).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "AVGE": {name: lp for lp, name in zip(torch.tensor(avges).tolist(), image_names)},
                                                            "AVGE_sk": {name: lp for lp, name in zip(torch.tensor(avges_sk).tolist(), image_names)}})

            with open(scene_dir + "/results_{}_mask.json".format(dataset), 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_{}_mask.json".format(dataset), 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
