"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from umi.common.timecode_util import mp4_get_start_datetime


from scripts_slam_pipeline.utils.constants import ARUCO_ID
from scripts_slam_pipeline.utils.misc import (
    get_single_path, custom_minimize, 
    plot_trajectories, plot_long_horizon_trajectory
)
from scripts_slam_pipeline.utils.data_loading import load_proprio_interp


def detect_turning_points(data):
    """
    Detect turning points where direction changes in the data.
    
    return timestamp, sign, position[timestamp]
    """
    position = [d[1] for d in data]
    timestamp = [d[0] for d in data]

    # remove stable area at beginning and end
    max_diff = np.max(position) - np.min(position)
    stable_diff = 0.1 * max_diff
    left = 0
    window_max, window_min = position[0], position[0]
    while window_max - window_min < stable_diff:
        left += 1
        window_max = max(window_max, position[left])
        window_min = min(window_min, position[left])

    right = -1
    window_max, window_min = position[-1], position[-1]
    while window_max - window_min < stable_diff:
        right -= 1
        window_max = max(window_max, position[right])
        window_min = min(window_min, position[right])
    
    position = position[left:right+1]
    timestamp = timestamp[left:right+1]

    # filter data
    # position = median_filter(position, size=5)
    position = gaussian_filter(position, sigma=1)

    turning_points = []
    for i in range(1, len(position) - 1):
        dt1 = position[i] - position[i - 1]
        dt2 = position[i + 1] - position[i]
        
        # Check for direction change (based on position delta sign change)
        if np.sign(dt1) != np.sign(dt2):
            turning_points.append((timestamp[i], np.sign(dt2)-np.sign(dt1), position[i]))
    
    return turning_points



# %%
@click.command()
@click.argument('session_dir')
@click.option('-c', '--calibration_dir', required=True, help='')
@click.option('--calibration_axis', type=str, default="x", help='')
@click.option('--init_offset', type=float, default=0)
def main(session_dir, calibration_dir, calibration_axis, init_offset):
    
    calibration_axis_index = {'x': 0, 'y': 1, 'z': 2}[calibration_axis]
    def get_axis_value(interp, value):
        try:
            pose = interp(value)
            return -pose[calibration_axis_index]
        except ValueError:
            return None

    calibration_dir = pathlib.Path(calibration_dir)

    session_dir = pathlib.Path(__file__).parent.joinpath(session_dir).absolute()
    latency_calib_dir = session_dir.joinpath('latency_calibration')

    # assert arcap_data_dir.is_dir()
    assert latency_calib_dir.is_dir()

    calibration_mp4 = get_single_path(
        list(latency_calib_dir.glob('**/*.MP4')) + list(latency_calib_dir.glob('**/*.mp4'))
    )


    print("load arcap trajectory")
    traj_interp, _ = load_proprio_interp(session_dir, latency=0.0, extend_boundary=10)
    print("detect_aruco")
    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco_newversion.py')
    assert script_path.is_file()
    
    camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
    aruco_config = calibration_dir.joinpath('aruco_config.yaml')
    assert camera_intrinsics.is_file()
    assert aruco_config.is_file()
    aruco_out_dir = latency_calib_dir.joinpath('tag_detection.pkl')
    if not aruco_out_dir.is_file():
        cmd = [
            'python', script_path,
            '--input', str(calibration_mp4),
            '--output', str(aruco_out_dir),
            '--intrinsics_json', camera_intrinsics,
            '--aruco_yaml', str(aruco_config),
            '--num_workers', '1'
        ]
        print(cmd)
        result = subprocess.run(cmd)
        assert result.returncode == 0
    else:
        print(f"tag_detection.pkl already exists, skipping {calibration_mp4}")


    print("align visual and trajectory")

    # get aruco trajectory
    video_start_time = mp4_get_start_datetime(str(calibration_mp4)).timestamp()

    aruco_pickle_path = latency_calib_dir.joinpath('tag_detection.pkl')
    with open(str(aruco_pickle_path), "rb") as fp:
        aruco_pkl = pickle.load(fp)
        aruco_trajectory = []
        for frame in aruco_pkl:
            if ARUCO_ID in frame['tag_dict']:
                x = frame['tag_dict'][ARUCO_ID]['tvec'][0]    # x-axis
                aruco_trajectory.append((frame['time']+video_start_time, x))

    
    # get arcap trajectory for plotting
    extend_range = 15
    timepoints = sorted([t for t, _ in aruco_trajectory])
    time_extend_before = np.linspace(timepoints[0]-extend_range, timepoints[0], 100).tolist()
    time_extend_after = np.linspace(timepoints[-1], timepoints[-1]+extend_range, 100).tolist()
    timepoints = time_extend_before + timepoints + time_extend_after

    arcap_trajectory_for_plotting = [
        (t, get_axis_value(traj_interp, t+init_offset)) 
        for t in timepoints
    ]
    plot_long_horizon_trajectory(
        aruco_trajectory, arcap_trajectory_for_plotting,
        title=f"Offset {init_offset:.2f} (move arcap {'left' if init_offset > 0 else 'right'})",
        save_dir=latency_calib_dir.joinpath(f'latency_trajectory_offset{init_offset}_long.pdf')
    )
    


    ###### algorithm 1: align by turning points
    turn_point_aruco = detect_turning_points(aruco_trajectory)
    turn_point_arcap = None

    for arcap_time_offset in [0.0, 1.0, -1.0, 2.0, -2.0]:
        arcap_time_offset += init_offset

        # get arcap trajectory
        arcap_trajectory = []
        timepoints = [t+arcap_time_offset for t, _ in aruco_trajectory]
        timepoints = sorted(timepoints)
        for t in timepoints:
            arcap_trajectory.append( ( t, get_axis_value(traj_interp, t) ) )
        turn_point_arcap = detect_turning_points(arcap_trajectory)
        
        # pickle.dump({
        #     "aruco": aruco_trajectory,
        #     "arcap": arcap_trajectory,
        # }, open(str(latency_calib_dir.joinpath(f'latency_data_{arcap_time_offset}.pkl')), 'wb'))
        
        plot_trajectories(aruco_trajectory, arcap_trajectory, 
                        [], [],
                        #turn_point_aruco, turn_point_arcap, 
                        latency_calib_dir.joinpath(f'latency_trajectory_offset{arcap_time_offset}.pdf'))

        if len(turn_point_aruco) == len(turn_point_arcap):
            break
        else:
            turn_point_arcap = None


    latency_of_arcap_result = {}
        
    if turn_point_arcap is not None:
        # algorithm 1 is good
        latency_of_arcap = []
        for (t1, d1, _), (t2, d2, _) in zip(turn_point_aruco, turn_point_arcap):
            if(d1 != d2):
                print("Direction mismatch")
                break
            latency_of_arcap.append(t2 - t1)
        else:
            if np.std(latency_of_arcap) < 0.1:
                # everything is good
                print(np.mean(latency_of_arcap), np.std(latency_of_arcap))

                latency_of_arcap_result = {
                    "mean": np.mean(latency_of_arcap),
                    "std": np.std(latency_of_arcap),
                    "data": latency_of_arcap,
                }
                
                lat = latency_of_arcap_result["mean"]
                calibrated_arcap_trajectory = []
                for t, v in arcap_trajectory:
                    calibrated_arcap_trajectory.append((t-lat, v))
                plot_trajectories(aruco_trajectory, calibrated_arcap_trajectory, 
                                [], [], 
                                latency_calib_dir.joinpath(f'latency_trajectory_final_algo_1.pdf'))
                
    

    #### algorithm 2: align by cross correlation

    print("switch to algorithm 2")

    aruco_trajectory = sorted(aruco_trajectory, key=lambda x: x[0])
    timepoints = [t for t, v in aruco_trajectory]
    aruco_pos = np.array([v for t, v in aruco_trajectory])
    aruco_pos = (aruco_pos - np.mean(aruco_pos)) / np.std(aruco_pos)  # normalized


    def mse_error(x):
        arcap_pos = np.array([get_axis_value(traj_interp, t+x[0]) for t in timepoints])
        arcap_pos = (arcap_pos - np.mean(arcap_pos)) / np.std(arcap_pos)
        return np.mean((aruco_pos - arcap_pos)**2)

    left_offset_bound  = -1.0 + init_offset
    right_offset_bound = 1.0 + init_offset
    epsilon = 0.01
    while True:
        res = custom_minimize(mse_error, 0.0, bounds=[(left_offset_bound, right_offset_bound)])
        if res.x - left_offset_bound < epsilon:
            left_offset_bound -= 1.0
            right_offset_bound -= 1.0
        elif right_offset_bound - res.x < epsilon:
            left_offset_bound += 1.0
            right_offset_bound += 1.0
        else:
            break

    print(res.x, mse_error(res.x))
        

    latency_of_arcap_result.update({
        "mean_2": res.x[0],
        "error": mse_error(res.x),
    })
    if "mean" not in latency_of_arcap_result:
        latency_of_arcap_result["mean"] = latency_of_arcap_result["mean_2"]
        
    with open(str(latency_calib_dir.joinpath('latency_of_arcap.json')), "w") as fp:
        json.dump(latency_of_arcap_result, fp)

    lat = latency_of_arcap_result["mean_2"]
    calibrated_arcap_trajectory = []
    for t, v in arcap_trajectory:
        calibrated_arcap_trajectory.append((t-lat, v))
    plot_trajectories(aruco_trajectory, calibrated_arcap_trajectory, 
                    [], [], 
                    latency_calib_dir.joinpath(f'latency_trajectory_final_algo_2.pdf'))


    
    




## %%
if __name__ == "__main__":
    main()
