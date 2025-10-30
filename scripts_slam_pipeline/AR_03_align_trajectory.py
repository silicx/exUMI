"""
python scripts_slam_pipeline/04_detect_aruco.py \
-i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room/demos \
-ci data_workspace/toss_objects/20231113/calibration/gopro_intrinsics_2_7k.json \
-ac data_workspace/toss_objects/20231113/calibration/aruco_config.yaml
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import json
import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import copy
import json
import av
from scipy.spatial.transform import Rotation as R
import yaml

from umi.common.timecode_util import mp4_get_start_datetime

from scripts_slam_pipeline.utils.dtact_sensor import Sensor
from scripts_slam_pipeline.utils.data_loading import (
    load_proprio_interp, load_tactile_interp, 
    save_frames_to_mp4_with_av,
)
from scripts_slam_pipeline.utils.misc import listOfDict_to_dictOfList
from scripts_slam_pipeline.utils.constants import (
    tx_arhand_inv,
    tx_arbase_at_flexivbase,
    tx_flexivobj_at_arobj,
    tx_flexivcamera_at_flexivobj,
)




# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-calib', '--arcap_latency_calibration_path', required=True, help='')
@click.option('-tactile_calib', '--tactile_calibration_path', required=True, help='')
@click.option('-n', '--num_workers', type=int, default=8)
def main(input_dir, arcap_latency_calibration_path, tactile_calibration_path, num_workers):

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')

    session_dir = input_dir.parent

    # read all trajectories
    latency = json.load(open(arcap_latency_calibration_path))["mean"]
    traj_interp, width_interp = load_proprio_interp(session_dir, latency, extend_boundary=0.5)
    tactile_interp = load_tactile_interp(session_dir, latency, extend_boundary=0.5)

    # tactile sensor class
    sensor_postprocess = {}
    with open(tactile_calibration_path, "r") as f:
        cfg_raw = yaml.load(f, Loader=yaml.FullLoader)
        for side in ["left", "right"]:
            cfg = copy.deepcopy(cfg_raw)
            cfg['sensor_id'] = side
            sensor_postprocess[side] = Sensor(cfg)



    fps = None
    n_frames_list = []
    for video_dir in tqdm(input_video_dirs, desc="check fps"):
        mp4_path = video_dir.joinpath('raw_video.mp4')
        with av.open(str(mp4_path), 'r') as container:
            stream = container.streams.video[0]
            n_frames = stream.frames
            if fps is None:
                fps = stream.average_rate
            else:
                if fps != stream.average_rate:
                    print(f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}")
                    exit(1)
            n_frames_list.append(n_frames)
            print(n_frames)


    def process_video(video_dir, n_frames):
        mp4_path = video_dir.joinpath('raw_video.mp4')

        # fetch meta data
        start_date = mp4_get_start_datetime(str(mp4_path))
        start_timestamp = start_date.timestamp()

        # align pose data
        aligned_proprio_data = []
        aligned_tactile_data = []

        
        drop_demo = False
        for i_frame in range(n_frames):
            timestamp = start_timestamp + i_frame / fps
            try:
                _pose = traj_interp(timestamp)
                _width = width_interp(timestamp)
            except ValueError as e:
                print(e)
                drop_demo = True
                break
            

            # post process of mocap
            _pose = np.array(_pose)
            assert _pose.shape == (7,), \
                f"Invalid pose shape: {_pose.shape}, expected (6,)"
            mat = np.identity(4)
            mat[:3, 3] = _pose[:3]
            mat[:3, :3] = R.from_quat(_pose[3:]).as_matrix()
            # get pose in flexivbase coordinates
            tx_obj = mat @ tx_arhand_inv
            mat = tx_arbase_at_flexivbase @ tx_obj @ tx_flexivobj_at_arobj @ tx_flexivcamera_at_flexivobj
            _pose[:3] = mat[:3, 3]
            _pose[3:] = R.from_matrix(mat[:3, :3]).as_quat()

            aligned_proprio_data.append( {
                'pose': _pose.tolist(),
                'width': float(_width),  # (1,) ->  float
            } )


            # post process of tactile sensor
            _tact_data = {}
            for side in ["left", "right"]:
                _sensor = sensor_postprocess[side]
                try:
                    raw_img = tactile_interp[side](timestamp)
                except ValueError as e:
                    print(e)
                    drop_demo = True
                    break


                if i_frame == 0:
                    _sensor.update_ref(raw_img)

                img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                img = _sensor.raw_image_2_xy_repr(img_GRAY)
                img = np.round(img).astype(np.uint8)
                img = _sensor.get_rectify_crop_image(img)


                _tact_data.update({
                    f"tactile_{side}": img,
                })
            
            if drop_demo:
                break
            
            aligned_tactile_data.append(_tact_data)

        if drop_demo:
            return False
        
        
        with open(video_dir.joinpath('aligned_arcap_poses.json'), 'w') as fp:
            proprio_data = listOfDict_to_dictOfList(aligned_proprio_data)
            json.dump(proprio_data, fp)


        tactile_data = listOfDict_to_dictOfList(aligned_tactile_data)
        for key, frames in tactile_data.items():
            save_frames_to_mp4_with_av(frames, video_dir.joinpath(f'{key}.mp4'), fps=int(fps))

        return True
    
    
    # sequential
    completed = []
    for vid_dir, n_frame in tqdm(zip(input_video_dirs, n_frames_list),
                                total=len(input_video_dirs), ncols=60):
        completed.append(process_video(vid_dir, n_frame))

    # parallel
    # with tqdm(total=len(input_video_dirs), ncols=60) as pbar:
    #     # one chunk per thread, therefore no synchronization needed
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = set()
    #         for vid_dir, n_frame in zip(input_video_dirs, n_frames_list):
    #             if len(futures) >= num_workers:
    #                 # limit number of inflight tasks
    #                 completed, futures = concurrent.futures.wait(futures, 
    #                     return_when=concurrent.futures.FIRST_COMPLETED)
    #                 pbar.update(len(completed))

    #             futures.add(executor.submit(process_video, 
    #                 vid_dir, n_frame))

    #         completed, futures = concurrent.futures.wait(futures)
    #         pbar.update(len(completed))


    print("Done!")
    num_of_skip = len([x for x in completed if not x])
    if num_of_skip > 0:
        print(f"Skipped {num_of_skip} demos from {len(input_video_dirs)}")

# %%
if __name__ == "__main__":
    main()
