import enum
import time
import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty



class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


class UsbCamera:
    def __init__(self, path, resolution, fps):
        self.cap = cv2.VideoCapture(path)

        w, h = resolution
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print("Cannot open camera", path)
            exit(-1)

    def get_data(self):
        tm = time.time()
        ret, frame = self.cap.read()
        
        assert ret
        return frame, tm

    def release(self):
        self.cap.release()



class CsiCamera:
    def __init__(self, path, resolution, fps):
        self.picamera = Picamera2()

        config = self.picamera.create_still_configuration(
            main={
                "format": 'RGB888',
                "size": resolution,
            })
        
        self.picamera.configure(config)
        self.picamera.start()

    
    def get_data(self):
        tm = time.time()
        frame = self.picamera.capture_array()
        return frame, tm

    def release(self):
        pass



class TactileCamera(mp.Process):
    """
    Call umi.common.usb_util.reset_all_elgato_devices
    if you are using Elgato capture cards.
    Required to workaround firmware bugs.
    """
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    
    def __init__(
            self,
            camera_cls,
            shm_manager: SharedMemoryManager,
            # v4l2 device file path
            # e.g. /dev/video0
            # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
            dev_video_path,
            resolution=(640, 480),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            get_max_k=120,
            receive_latency=0.0,
            cap_buffer_size=1,
            num_threads=2,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        
        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = {
            'color': np.empty(
                shape=shape+(3,), dtype=np.uint8)
        }
        # examples['camera_capture_timestamp'] = 0.0
        # examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        ring_buffer = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=get_max_k
            # get_max_k=get_max_k,
            # get_time_budget=0.2,
            # put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create video recorder
        
        # default to nvenc GPU encoder
        # video_recorder = VideoRecorder.create_hevc_nvenc(
        #     shm_manager=shm_manager,
        #     fps=capture_fps, 
        #     input_pix_fmt='bgr24', 
        #     bit_rate=6000*1000)
        # assert video_recorder.fps == capture_fps

        # copied variables
        self.camera_cls = camera_cls
        self.shm_manager = shm_manager
        self.dev_video_path = dev_video_path
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        # self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        # data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        # self.video_recorder.start(
        #     shm_manager=self.shm_manager, 
        #     data_example=data_example)
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        # self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        # self.video_recorder.start_wait()
    
    def end_wait(self):
        self.join()
        # self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    @property
    def queue_count(self) -> int:
        return self.ring_buffer.qsize()  #count

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        # else:
        #     return self.ring_buffer.get_last_k(k, out=out)
    
    def get_all(self):
        data = self.ring_buffer.get_all()
        return data

    def start_recording(self, video_path: str, start_time: float=-1):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(self.num_threads)
        # cv2.setNumThreads(self.num_threads)

    
        # open VideoCapture
        # cap = cv2.VideoCapture(self.dev_video_path, cv2.CAP_V4L2)
        cap = self.camera_cls(self.dev_video_path, self.resolution, self.capture_fps)

        try:
            w, h = self.resolution
            fps = self.capture_fps
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            # # set fps
            # cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
            # cap.set(cv2.CAP_PROP_FPS, fps)

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                iter_start_time = time.monotonic()
                
                # ret = cap.grab()
                # assert ret
                
                # # directly write into shared memory to avoid copy
                # frame = self.video_recorder.get_img_buffer()
                # ret, frame = cap.retrieve(frame)
                # t_recv = time.time()
                # assert ret
                # mt_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                # t_cap = mt_cap - time.monotonic() + time.time()
                # t_cal = t_recv - self.receive_latency # calibrated latency

                # record frame
                # if self.video_recorder.is_ready():
                #     self.video_recorder.write_img_buffer(frame, frame_time=t_cal)
                frame, t_cal = cap.get_data()
                

                data = dict()
                # data['camera_receive_timestamp'] = t_recv
                # data['camera_capture_timestamp'] = t_cap
                data['color'] = frame
                
                
                put_data = data

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[t_cal],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data) # , wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = t_cal
                    self.ring_buffer.put(put_data) # , wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                    

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    # print(f'[UvcCamera {self.dev_video_path}] FPS {frequency}')
                    pass


                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        # self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        pass
                        # self.video_recorder.stop_recording()

                iter_idx += 1

                iter_duration = time.monotonic() - iter_start_time
                if iter_duration < 1./fps:
                    time.sleep(1./fps-iter_duration)


        finally:
            # self.video_recorder.stop()
            # When everything done, release the capture
            cap.release()

