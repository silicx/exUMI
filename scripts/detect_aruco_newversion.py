# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from tqdm import tqdm
import yaml
import json
import av
import numpy as np
import cv2
from cv2 import aruco
import pickle
from typing import Dict

from umi.common.pose_util import *
from umi.common.cv_util import (
    parse_aruco_config, 
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    draw_predefined_mask
)

def draw_non_gripper_mask(img, color=(0,0,0), mirror=True, gripper=True, finger=True, use_aa=False):
    h, w = img.shape[:2]
    fill_height = int(h * 2 / 3)
    cv2.rectangle(img, (0, 0), (w, fill_height), color, -1)
    return img




# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def my_detect_localize_aruco_tags(
        img: np.ndarray, 
        aruco_dict: aruco.Dictionary, 
        marker_size_map: Dict[int, float], 
        fisheye_intr_dict: Dict[str, np.ndarray], 
        refine_subpix: bool=True,
        non_fisheye_lens: bool=False):
    K = fisheye_intr_dict['K']
    if not non_fisheye_lens:
        D = fisheye_intr_dict['D']
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    if len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        
        marker_size_m = marker_size_map[this_id]
        if not non_fisheye_lens:
            undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        else:
            undistorted = this_corners
        rvec, tvec, markerPoints = my_estimatePoseSingleMarkers(
            undistorted, marker_size_m, K, np.zeros((1,5)))
        

        tag_dict[this_id] = {
            'rvec': np.array(rvec).squeeze(),
            'tvec': np.array(tvec).squeeze(),
            'corners': this_corners.squeeze(),
        }
    return tag_dict




# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
@click.option('-msk', '--non_gripper_mask', is_flag=True, default=False, help='Mask top 2/3 region, since we are only interested in gripper')
def main(input, output, intrinsics_json, aruco_yaml, num_workers, non_gripper_mask):
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))

    results = list()
    print(os.path.expanduser(input))
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.height, in_stream.width])[::-1]
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)

        for i, frame in enumerate(in_container.decode(in_stream)):
            img = frame.to_ndarray(format='rgb24')
            frame_cts_sec = frame.pts * in_stream.time_base
            # avoid detecting tags in the mirror
            # img = draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=False, finger=False)

            if non_gripper_mask:
                img = draw_non_gripper_mask(img)


            tag_dict = my_detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True
            )
            result = {
                'frame_idx': i,
                'time': float(frame_cts_sec),
                'tag_dict': tag_dict
            }
            results.append(result)
    
    # dump
    pickle.dump(results, open(os.path.expanduser(output), 'wb'))

# %%
if __name__ == "__main__":
    main()
