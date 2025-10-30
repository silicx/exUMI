import h5py
import numpy as np
import json
import re
from typing import cast
import av
from tqdm import tqdm
import pathlib
import time
from scipy.interpolate import interp1d
import time
import cv2

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

from scripts_slam_pipeline.utils.misc import get_single_path

__all__ = [
    "load_proprio_interp",
    "load_tactile_interp",
    "load_realsense_interp",
    "save_frames_to_mp4_with_av",
    "load_realsense_rgb_data",
]



########################################### interpolator ###########################################


class WrappedInterp1d:
    def __init__(self, X, Y, extend_boundary=None):
        """interpolated matching from X to Y
        not necessary to sort the data"""

        sort_id = np.argsort(X)
        X = np.array(X)[sort_id]
        Y = [Y[i] for i in sort_id]
        
        self.traj_interp = interp1d(X, Y, axis=0, bounds_error=True)

        extend_boundary = 0 if extend_boundary is None else extend_boundary
        self.left_min, self.left_max = X[0]-extend_boundary, X[0]
        self.right_min, self.right_max = X[-1], X[-1]+extend_boundary
        self.left_val = Y[0]
        self.right_val = Y[-1]

    def __call__(self, x):
        if self.left_min < x < self.left_max:
            print(f"WARNING: extrapolating left: {self.left_min} < {x} < {self.left_max}")
            return self.left_val
        elif self.right_min < x < self.right_max:
            print(f"WARNING: extrapolating right: {self.right_min} < {x} < {self.right_max}")
            return self.right_val
        else:
            return self.traj_interp(x)
        

class ImageLinearInterp1d:
    def __init__(self, X, Y, extend_boundary=None):
        """interpolated matching from X to Y
        not necessary to sort the data"""

        sort_id = np.argsort(X)
        X = [X[i] for i in sort_id]
        Y = [Y[i] for i in sort_id]

        self.X = [X[0]-extend_boundary] + X + [X[-1]+extend_boundary]
        self.Y = [Y[0]] + Y + [Y[-1]]


    def __call__(self, val):
        if val == self.X[-1]:
            return self.Y[-1]
        else:
            right = np.searchsorted(self.X, val, side='right')  # a[i-1] <= v < a[i]

            if right <= 0 or right >= len(self.X):
                raise ValueError(f"extrapolating: {val} out of range {self.X[0]} - {self.X[-1]}")
            else:
                x0, x1 = self.X[right-1], self.X[right]
                y0, y1 = self.Y[right-1], self.Y[right]
                
                dtype = y0.dtype
                assert dtype in [np.uint8, np.uint16], f"Invalid dtype: {dtype}, expected uint8 or uint16"

                y0, y1 = y0.astype(float), y1.astype(float)
                res = y0 + (y1-y0) * (val-x0) / (x1-x0)
                res = np.round(res).astype(dtype)
                return res
            

####################################### interpolator loader ########################################

def load_proprio_interp(session_dir, latency, extend_boundary=None, rotation_minimum=100):
    """rotation_minimum: minimal possible rotation value. as5600 readings that smaller than this value will be added by 4096"""

    with open("ARCap/as5600_calib/gripper_calibration.json", "r") as fp:
        width_calib = json.load(fp)["calibration"]
        widths, values = zip(*width_calib)
        
        jump_point = None
        for i in range(len(values)-1):
            if np.abs(values[i] - values[i+1]) > 2048:
                # here's a jump in the circular data
                if jump_point is not None:
                    raise ValueError(f"Multiple jumps found in calibration data: {values}")
                jump_point = i

        if jump_point is None:
            as5600_preprocess = lambda x: x
        else:
            group_1, group_2 = values[:jump_point+1], values[jump_point+1:]
            max_1, min_1 = np.max(group_1), np.min(group_1)
            max_2, min_2 = np.max(group_2), np.min(group_2)

            if min_1 > max_2:
                threshold = (min_1 + max_2) / 2
            elif min_2 > max_1:
                threshold = (min_2 + max_1) / 2
            else:
                raise ValueError(f"????why the two groups overlap???? {values}")
            
            def as5600_preprocess(x):
                if x < threshold:
                    return x + 4096
                else:
                    return x

        values = list(map(as5600_preprocess, values))
        if values[0] > values[-1]:
            values = [values[0]+200] + values + [values[-1]-200]
        else:
            values = [values[0]-200] + values + [values[-1]+200]
        widths = list(widths)
        widths = [widths[0]] + widths + [widths[-1]]
        as5600_to_width = WrappedInterp1d(values, widths, extend_boundary=10)


    time_list = []
    width_list = []
    pose_list = []

    all_proprio_files = list(pathlib.Path(session_dir).glob('tactile_*/angle_*.json'))
    if len(all_proprio_files) == 0:
        raise OSError(f"No proprio data found in {session_dir}")
    
    for proprio_chunk_file in all_proprio_files:
        with open(str(proprio_chunk_file), "r") as fp:
            proprio_data = json.load(fp)

            _times  = proprio_data["timestamps"]
            _angles = [x[0] for x in proprio_data["angles"]]   # "angles": [[4063], [4063], [4063], [4063], ...]
            _pose   = proprio_data["data"]
            assert len(_angles) == len(_times) == len(_pose), f"Data length mismatch: {_angles} vs {_times} vs {_pose}"

            time_list += _times
            width_list += [
                as5600_to_width(as5600_preprocess(x))
                for x in _angles]
            pose_list += _pose

    time_list = np.array(time_list)

    width_interp = WrappedInterp1d(time_list - latency, width_list, extend_boundary=extend_boundary)
    pose_interp  = WrappedInterp1d(time_list - latency, pose_list, extend_boundary=extend_boundary)
    print("proprio time range", width_interp.left_max, width_interp.right_min)

    return pose_interp, width_interp


def load_tactile_interp(session_dir, latency, extend_boundary=None):

    tactile_interp_dict = {}


    for side in ["left", "right"]:
        tactile_data = []

        all_tactile_files = list(pathlib.Path(session_dir).glob(f'tactile_*/{side}_*.json'))
        if len(all_tactile_files) == 0:
            raise OSError(f"No tactile data found in {session_dir}")
        
        start_time = time.time()

        for tactile_meta_file in all_tactile_files:
            with open(str(tactile_meta_file), "r") as fp:
                # time

                try:
                    timestamp_list = json.load(fp)
                except json.decoder.JSONDecodeError as e:
                    print(str(tactile_meta_file))
                    raise e


                # tactile frame
                tactile_video_file = tactile_meta_file.with_suffix(".mp4")
                raw_frames = load_frames_from_mp4(tactile_video_file)
                assert len(raw_frames) == len(timestamp_list), f"Frame count mismatch: {len(raw_frames)} vs {len(timestamp_list)}, {tactile_video_file}"

            tactile_data += list(zip(timestamp_list, raw_frames))
        
        tactile_interp_dict[side] = ImageLinearInterp1d([x[0]-latency for x in tactile_data], [x[1] for x in tactile_data], extend_boundary=extend_boundary)

        print(f"Loaded {len(tactile_data)} tactile frames for {side}, in {time.time()-start_time:.2f} sec")
        print("tactile valid time range", tactile_interp_dict[side].X[0], tactile_interp_dict[side].X[-1])

    return tactile_interp_dict




def load_realsense_interp(session_dir, latency, extend_boundary=None):

    realsense_dir = get_single_path(pathlib.Path(session_dir).glob("realsense_*"))

    realsense_timestamps = []
    realsense_frames = {
        "colored": [],
        "depth": [],
    }

    start_time = time.time()

    for tstamp_file in realsense_dir.glob("realsense_*.json"):
        with open(str(tstamp_file), "r") as fp:
            tstamp = json.load(fp)
        m = re.match(r"realsense_(\d+).json", tstamp_file.name)
        if m is None:
            raise ValueError(f"Invalid file name: {tstamp_file.name}")
        file_id = m.group(1)

        realsense_timestamps.extend([t / 1000. for t in tstamp])  # ms -> s

        # mode = colored
        mode = "colored"
        realsense_file = realsense_dir.joinpath(f"realsense_{mode}_{file_id}.mp4")
        frames = load_frames_from_mp4(realsense_file)
        assert len(frames) == len(tstamp), \
            f"Frame count mismatch: {len(frames)} vs {len(tstamp)}, {realsense_file}"
        realsense_frames[mode].extend(frames)

        mode = "depth"
        realsense_file = realsense_dir.joinpath(f"realsense_{mode}_{file_id}.h5")
        frames = load_frames_from_h5py(realsense_file, mode=mode)
        assert len(frames) == len(tstamp), \
            f"Frame count mismatch: {len(frames)} vs {len(tstamp)}, {realsense_file}"
        realsense_frames[mode].extend(frames)



    realsense_interp_dict = {}
    for mode in ["colored", "depth"]:
        realsense_interp_dict[mode] = ImageLinearInterp1d(
            [x-latency for x in realsense_timestamps], realsense_frames[mode], 
            extend_boundary=extend_boundary)

    print(f"Loaded {len(realsense_timestamps)} tactile frames, in {time.time()-start_time:.2f} sec")

    return realsense_interp_dict



###################################### file loader and saver #######################################


def load_frames_from_h5py(h5py_file, mode="colored"):
    assert mode in ["colored", "depth"], f"Invalid mode: {mode}, expected 'colored' or 'depth'"

    key = {
        "colored": "colored_frames",
        "depth": "depth_frames",
    }[mode]
    
    with h5py.File(h5py_file, 'r') as f:
        assert key in f.keys(), f"Key {key} not found in {h5py_file}"
        frames = f[key][...]
        frames = cast(np.ndarray, frames)
        if mode == "depth":
            assert len(frames.shape) == 3, f"Invalid frame shape: {frames.shape}, expected (N, H, W)"
            assert frames.dtype == np.uint16, f"Invalid frame dtype: {frames.dtype}, expected uint16"
        else:
            assert len(frames.shape) == 4, f"Invalid frame shape: {frames.shape}, expected (N, H, W, C)"
            assert frames.dtype == np.uint8, f"Invalid frame dtype: {frames.dtype}, expected uint8"
            assert frames.shape[3] == 3, f"Invalid frame channel count: {frames.shape[3]}, expected 3"
            
        return [frames[i] for i in range(frames.shape[0])]


def load_realsense_rgb_data(realsense_dir, aruco_trajectory, init_offset, extend_range):
    realsense_timestamp_and_file_id = []
    for tstamp_file in realsense_dir.glob("realsense_*.json"):
        with open(str(tstamp_file), "r") as fp:
            tstamp = json.load(fp)
        m = re.match(r"realsense_(\d+).json", tstamp_file.name)
        if m is None:
            raise ValueError(f"Invalid file name: {tstamp_file.name}")
        file_id = int(m.group(1))
        realsense_timestamp_and_file_id.extend([[t / 1000., file_id] for t in tstamp])  # ms -> s
    
    aruco_timepoints = [t+init_offset for t, _ in aruco_trajectory]
    start_time = np.min(aruco_timepoints)-extend_range
    end_time = np.max(aruco_timepoints)+extend_range
    # filter the timestamps
    realsense_files_involved = {
        f for t, f in realsense_timestamp_and_file_id
        if start_time <= t <= end_time
    }  # realsense files that are involved in the time range of GoPro aruco

    print("GoPro time range", start_time, "to", end_time)
    tstamps = [t for t, f in realsense_timestamp_and_file_id]
    print("Realsense time range", np.min(tstamps), "to", np.max(tstamps))
    print("Involved realsense files:", min(realsense_files_involved), "to", max(realsense_files_involved))

    # load the realsense data
    realsense_timestamp = []
    realsense_frames = []
    for file_id in tqdm(realsense_files_involved, ncols=60):
        json_file = realsense_dir.joinpath(f"realsense_{file_id}.json")
        rgb_file = realsense_dir.joinpath(f"realsense_colored_{file_id}.mp4")
        with open(str(json_file), "r") as fp:
            tstamp = json.load(fp)

        frames = load_frames_from_mp4(rgb_file)

        realsense_timestamp.extend([t / 1000. for t in tstamp])  # ms -> s
        realsense_frames.extend(frames)

    assert len(realsense_timestamp) == len(realsense_frames), \
        f"Timestamp and frames length mismatch, {len(realsense_timestamp)}, {len(realsense_frames)}"

    print("Realsense time range", np.min(realsense_timestamp), "to", np.max(realsense_timestamp))

    return realsense_timestamp, realsense_frames




def load_frames_from_mp4(mp4_path):
    video_capture = cv2.VideoCapture(str(mp4_path))
    assert video_capture.isOpened(), f"Cannot open video file: {mp4_path}"
    raw_frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        raw_frames.append(frame)
    video_capture.release()
    return raw_frames

def save_frames_to_mp4(frames, mp4_path, fps=60):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()


def save_frames_to_h5py(frames, h5_path):
    frames = np.array(frames)
    with h5py.File(str(h5_path), 'w') as f:
        f.create_dataset("frames", data=frames) #, compression="gzip")


def save_frames_to_mp4_with_av(frames, mp4_path, fps=60):

    h, w, _ = frames[0].shape
    
    with av.open(str(mp4_path), mode='w') as container:
        stream = container.add_stream('h264', rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = 'yuv420p'

        for i, frame in enumerate(frames):
            av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packet = stream.encode(av_frame)
            container.mux(packet)
                
        packet = stream.encode(None)
        container.mux(packet)