# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import h5py
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from scipy.signal import medfilt, savgol_filter
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl, Zlib
register_codecs()



def filter_xyz(eef_pos):
    outlier_threshold = 0.2
    outlier_kernel_size = 11
    smoothing_kernel_size = 5
    
    for i in range(3):
        axis = eef_pos[:, i]
        filtered = medfilt(axis, kernel_size=outlier_kernel_size)
        max_vel = np.abs(filtered - axis)

        axis = np.where(
            max_vel>outlier_threshold,
            filtered,  # outliers, replace with filtered
            axis.copy() # keep original
        )
        
        axis = savgol_filter(axis, smoothing_kernel_size, 2)
        eef_pos[:, i] = axis

        
    return eef_pos



# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-otr', '--out_tactile_res', type=str, default='224,224')
@click.option('-orr', '--out_realsense_res', type=str, default='720,1280')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('--num_sample_per_plan', type=int, default=None)
@click.option('-gth', '--gripper_threshold', type=float, default=None, help="")
@click.option('--realsense', is_flag=True, default=False)
def main(input, output, out_res, out_tactile_res, out_realsense_res, out_fov, compression_level, 
         num_workers, num_sample_per_plan, gripper_threshold, realsense):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))
    out_tactile_res = tuple(int(x) for x in out_tactile_res.split(','))
    out_realsense_res = tuple(int(x) for x in out_realsense_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))

        if num_sample_per_plan is not None:
            assert num_sample_per_plan <= len(plan), f"num_sample_per_plan must <= len(plan) = {len(plan)} ({ipath})"
            plan = plan[:num_sample_per_plan]
        
        videos_dict = defaultdict(list)
        for plan_episode in plan:
            grippers = plan_episode['grippers']
            
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
                
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):    
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]

                eef_pos = filter_xyz(eef_pos)

                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']

                if gripper_threshold:
                    max_w = 0.10
                    min_w = 0.02
                    bin_width = (gripper_widths > gripper_threshold).astype(np.float32)
                    gripper_widths = bin_width * (max_w - min_w) + min_w

                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
            
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    depth_compressor = Zlib(level=compression_level)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    # tactile camera
    for name in ["tactile_combined"]:
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_tactile_res + (3,),
            chunks=(1,) + out_tactile_res + (1,),
            compressor=img_compressor,
            dtype=np.uint8
            )
    if realsense:
        _ = out_replay_buffer.data.require_dataset(
            name="realsense_colored",
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_realsense_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
        _ = out_replay_buffer.data.require_dataset(
            name="realsense_depth",
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_realsense_res,
            chunks=(1,) + out_realsense_res,
            compressor=depth_compressor,
            dtype=np.uint16
        )


    def video_to_zarr(replay_buffer, mp4_path, tasks):
        # pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        # tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            # for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, ncols=70, leave=False):
            for frame_idx, frame in enumerate(container.decode(in_stream)):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False

        # for tactile camera
        mp4_path = pathlib.Path(mp4_path)
        tactile_left_path = mp4_path.parent.joinpath("tactile_left.mp4")
        tactile_right_path = mp4_path.parent.joinpath("tactile_right.mp4")
        img_array = replay_buffer.data["tactile_combined"]

        curr_task_idx = 0
        with av.open(str(tactile_left_path)) as container_left, \
                av.open(str(tactile_right_path)) as container_right:
            in_stream_left = container_left.streams.video[0]
            in_stream_left.thread_count = 1
            in_stream_right = container_right.streams.video[0]
            in_stream_right.thread_count = 1

            buffer_idx = 0
            for frame_idx, (frame_left, frame_right) in enumerate(zip(
                container_left.decode(in_stream_left),
                container_right.decode(in_stream_right)
            )):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img_left = frame_left.to_ndarray(format='rgb24')
                    img_right = frame_right.to_ndarray(format='rgb24')
                    img_left = cv2.rotate(img_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_right = cv2.rotate(img_right, cv2.ROTATE_90_CLOCKWISE)
                    img = cv2.hconcat([img_left, img_right])
                    img = cv2.resize(img, out_tactile_res, interpolation=cv2.INTER_AREA)

                    assert img.shape[:2] == out_tactile_res, f"img.shape: {img.shape}, out_tactile_res: {out_tactile_res}"

                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False


        # for realsense rgb and depth
        if realsense:
            key = "realsense_colored"
            tactile_path = mp4_path.parent.joinpath(key + ".mp4")
            img_array = replay_buffer.data[key]

            curr_task_idx = 0
            with av.open(str(tactile_path)) as container:
                in_stream = container.streams.video[0]
                in_stream.thread_count = 1
                buffer_idx = 0
                for frame_idx, frame in enumerate(container.decode(in_stream)):
                    if curr_task_idx >= len(tasks):
                        # all tasks done
                        break
                    
                    if frame_idx < tasks[curr_task_idx]['frame_start']:
                        # current task not started
                        continue
                    elif frame_idx < tasks[curr_task_idx]['frame_end']:
                        if frame_idx == tasks[curr_task_idx]['frame_start']:
                            buffer_idx = tasks[curr_task_idx]['buffer_start']
                        
                        # do current task
                        img = frame.to_ndarray(format='rgb24')
                        img_array[buffer_idx] = img
                        buffer_idx += 1
                        
                        if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                            # current task done, advance
                            curr_task_idx += 1
                    else:
                        assert False

            depth_path = mp4_path.parent.joinpath("realsense_depth.h5")
            img_array = replay_buffer.data["realsense_depth"]

            curr_task_idx = 0
            buffer_idx = 0
            with h5py.File(str(depth_path), 'r') as fp:
                array_depth = fp['frames'][...]
                for frame_idx in range(array_depth.shape[0]):
                    frame = array_depth[frame_idx]
                    if curr_task_idx >= len(tasks):
                        # all tasks done
                        break
                    
                    if frame_idx < tasks[curr_task_idx]['frame_start']:
                        # current task not started
                        continue
                    elif frame_idx < tasks[curr_task_idx]['frame_end']:
                        if frame_idx == tasks[curr_task_idx]['frame_start']:
                            buffer_idx = tasks[curr_task_idx]['buffer_start']
                        
                        img_array[buffer_idx] = frame
                        buffer_idx += 1
                        
                        if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                            # current task done, advance
                            curr_task_idx += 1
                    else:
                        assert False
                    
                    
    with tqdm(total=len(vid_args), ncols=60) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print("Dumped rgb camera")
    print([x.result() for x in completed])

    



    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()
