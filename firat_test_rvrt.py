# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader

from models.network_rvrt import RVRT as net
from utils import utils_image as util
from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, SingleVideoRecurrentTestDataset, Mp4Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='001_RVRT_videosr_bi_REDS_30frames', help='tasks: 001 to 006')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--fname_lq', type=str, default='testsets/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[100,128,128],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in data loading')
    parser.add_argument('--folder_out', type=str, default='results', help='path to save results.')
    args = parser.parse_args()

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)

    save_dir = args.folder_out
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)

    freq_log = 500
    num_frames_at_a_time = 15_000
    total_frames = 85_130
    l_frame_intervals = np.linspace(0, 85130, int(np.ceil(total_frames/num_frames_at_a_time))+1, dtype=int)

    for i_fr in range(len(l_frame_intervals)-1): 
        frame_start = l_frame_intervals[i_fr]
        frame_end = l_frame_intervals[i_fr+1]
        print(f'Starting with frame intervals {frame_start} to {frame_end}.')

        test_set = Mp4Generator(fname_in=args.fname_lq, frame_start=frame_start, frame_end=frame_end)
        test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)

        assert len(test_loader) != 0, f'No dataset found at {args.fname_lq}'

        for idx, batch in enumerate(test_loader):
            lq = batch['L'].to(device)
            
            # inference
            with torch.no_grad():
                output = test_video(lq, model, args)


            for i in range(output.shape[1]):
                # save image
                img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img.ndim == 3:
                    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                frame_id = batch['lq_path'][i]
                cv2.imwrite(f'{save_dir}/{frame_id:010d}.jpg', img)\
                    
                if (idx + frame_start) % freq_log == 0:
                    print(f'finished exporting {idx + frame_start}/{frame_end} of the frames.')
                                    
            
                print('Testing ({:2d}/{})'.format(idx, len(test_loader)))
        print(f'finished exporting {frame_end}/{total_frames} of the frames.')
        del test_set

    

def prepare_model_dataset(args):
    ''' prepare model and dataset according to args.task. '''

    # define model
    if args.task == '001_RVRT_videosr_bi_REDS_30frames':
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['002_RVRT_videosr_bi_Vimeo_14frames', '003_RVRT_videosr_bd_Vimeo_14frames']:
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['004_RVRT_videodeblurring_DVD_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['005_RVRT_videodeblurring_GoPro_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task == '006_RVRT_videodenoising_DAVIS_16frames':
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], nonblind_denoising=True, cpu_cache_length=100)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = True

    # download model
    model_path = f'model_zoo/rvrt/{args.task}.pth'
    if os.path.exists(model_path):
        print(f'loading model from ./model_zoo/rvrt/{model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/RVRT/releases/download/v0.0/{}'.format(os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)

    # download datasets
    if os.path.exists(f'{args.folder_lq}'):
        print(f'using dataset from {args.fname_lq}')
    else:
        if 'vimeo' in args.fname_lq.lower():
            print(f'Vimeo dataset is not at {args.fname_lq}! Please refer to #training of Readme.md to download it.')
        else:
            os.makedirs('testsets', exist_ok=True)
            for dataset in datasets:
                url = f'https://github.com/JingyunLiang/VRT/releases/download/v0.0/testset_{dataset}.tar.gz'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading testing dataset {dataset}')
                open(f'testsets/{dataset}.tar.gz', 'wb').write(r.content)
                os.system(f'tar -xvf testsets/{dataset}.tar.gz -C testsets')
                os.system(f'rm testsets/{dataset}.tar.gz')

    return model


def test_video(lq, model, args):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = args.tile[0]
        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = args.scale
            num_frame_overlapping = args.tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if args.nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, args)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = args.window_size
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' test the clip as a whole or as patches. '''

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output


if __name__ == '__main__':
    main()
