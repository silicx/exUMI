"""
python scripts_slam_pipeline/06_generate_dataset_plan.py -i data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
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
import pickle
import numpy as np
import json
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import av
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.pose_util import pose_to_mat, mat_to_pose


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Project directory')
@click.option('-o', '--output', default=None)
@click.option('-to', '--tcp_offset', type=float, default=0.089, help="Distance from gripper tip to mounting screw")
@click.option('--ignore_cameras', type=str, default=None, help="comma separated string of camera serials to ignore")
@click.option('-gth', '--gripper_threshold', type=float, default=None, help="")
@click.option('--check_realsense', is_flag=True, default=False)
def main(input, output, tcp_offset, ignore_cameras, gripper_threshold, check_realsense):
    # %% stage 0
    # gather inputs
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath('demos')
    if output is None:
        output = input_path.joinpath('dataset_plan.pkl')

    trajectory_file = "aligned_arcap_poses.json"

    # tcp to camera transform
    # all unit in meters
    # y axis in camera frame
    cam_to_center_height = 0.080 # constant for UMI
    # optical center to mounting screw, positive is when optical center is in front of the mount
    cam_to_mount_offset = 0.01465 # constant for GoPro Hero 9,10,11
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset

    pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0,0,0])
    tx_cam_tcp = pose_to_mat(pose_cam_tcp)


    
    # %% stage 1
    # loop over all demo directory to extract video metadata
    # output: video_meta_df
    
    # find videos
    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])

    # ignore camera
    ignore_cam_serials = set()
    if ignore_cameras is not None:
        serials = ignore_cameras.split(',')
        ignore_cam_serials = set(serials)
    
    fps = None
    video_meta = list()
    with ExifToolHelper() as et:
        for video_dir in video_dirs:            
            mp4_path = video_dir.joinpath('raw_video.mp4')
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta['QuickTime:CameraSerialNumber']
            start_date = mp4_get_start_datetime(str(mp4_path))
            start_timestamp = start_date.timestamp()

            if cam_serial in ignore_cam_serials:
                print(f"Ignored {video_dir.name}")
                continue
            
            csv_path = video_dir.joinpath(trajectory_file)
            print(csv_path)
            if not csv_path.is_file():
                print(f"Ignored {video_dir.name}, no {trajectory_file}")
                continue
            
            
            with av.open(str(mp4_path), 'r') as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                if fps is None:
                    fps = stream.average_rate
                else:
                    if fps != stream.average_rate:
                        print(f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}")
                        exit(1)
            duration_sec = float(n_frames / fps)
            end_timestamp = start_timestamp + duration_sec
            
            video_meta.append({
                'video_dir': video_dir,
                'camera_serial': cam_serial,
                'start_date': start_date,
                'n_frames': n_frames,
                'fps': fps,
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp
            })

    if len(video_meta) == 0:
        print("No valid videos found!")
        exit(1)

    
    # %% stage 6
    # generate dataset plan
    # output
    # all_plans = [{
    #     "episode_timestamps": np.ndarray,
    #     "grippers": [{
    #         "tcp_pose": np.ndarray,
    #         "gripper_width": np.ndarray
    #     }],
    #     "cameras": [{
    #         "video_path": str,
    #         "video_start_end": Tuple[int,int]
    #     }]
    # }]
    all_plans = list()
    for demo_idx, demo_data in enumerate(video_meta):
        start_timestamp = demo_data['start_timestamp']
        end_timestamp = demo_data['end_timestamp']
        dt = 1 / demo_data['fps']

        # descritize timestamps for all videos
        n_frames = demo_data['n_frames']
        demo_timestamps = np.arange(n_frames) * float(dt) + start_timestamp

        # load pose and gripper data for each video
        
        video_dir = demo_data['video_dir']

        # check realsense data
        if check_realsense:
            if not video_dir.joinpath('realsense_colored.mp4').is_file():
                print(f"Realsense video not found in {video_dir.name}, skipping")
                continue
            if not video_dir.joinpath('realsense_depth.h5').is_file():
                print(f"Realsense depth not found in {video_dir.name}, skipping")
                continue
            

        # load SLAM data
        json_path = video_dir.joinpath(trajectory_file)
        assert json_path.is_file()

        proprio_data = json.load(open(json_path, 'r'))
        cam_7dpose = np.array(proprio_data['pose'])
        cam_pos = cam_7dpose[:, :3]
        cam_rot_quat_xyzw = cam_7dpose[:, 3:]
        cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
        cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
        cam_pose[:,3,3] = 1
        cam_pose[:,:3,3] = cam_pos
        cam_pose[:,:3,:3] = cam_rot.as_matrix()
        tx_slam_cam = cam_pose
        # tx_tag_cam = tx_tag_slam @ tx_slam_cam
        tx_tag_cam = tx_slam_cam
        # transform to tcp frame
        tx_tag_tcp = tx_tag_cam @ tx_cam_tcp
        pose_tag_tcp = mat_to_pose(tx_tag_tcp)

        # get gripper data
        this_gripper_widths = np.array(proprio_data['width'])
            
        # output value
        assert len(pose_tag_tcp) == n_frames, f"{len(pose_tag_tcp)} != {n_frames}"
        assert len(this_gripper_widths) == n_frames, f"{len(this_gripper_widths)} != {n_frames}"
        all_cam_poses = pose_tag_tcp
        all_gripper_widths = this_gripper_widths

        width = all_gripper_widths
        if gripper_threshold:
            max_w = np.max(width)
            min_w = 0.02
            bin_width = (width > gripper_threshold).astype(np.float32)
            width = bin_width * (max_w - min_w) + min_w
                    
        # gripper cam
        grippers = {
            "tcp_pose": all_cam_poses,
            "gripper_width": width,
            "demo_start_pose": all_cam_poses[0],
            "demo_end_pose": all_cam_poses[-1],
        }

        # all cams
        cameras = {
            "video_path": str(video_dir.joinpath('raw_video.mp4').relative_to(video_dir.parent)),
            "video_start_end": (0, n_frames)
        }
            
        all_plans.append({
            "episode_timestamps": demo_timestamps,
            "grippers": [grippers],
            "cameras": [cameras]
        })

    # dump the plan to pickle
    pickle.dump(all_plans, output.open('wb'))
    


## %%
if __name__ == "__main__":
    main()
