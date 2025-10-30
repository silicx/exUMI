import socket
import time
from argparse import ArgumentParser
import pybullet as pb
import numpy as np
from ip_config import *
from quest_robot_module_clean import QuestRightArmLeapModule
from scipy.spatial.transform import Rotation as R



class DataChunker:
    def __init__(self, chunksize=1200):
        self.chunksize = chunksize
        self.records = []
        self.timestamps = []

        self.last_save_dir = None

    def put(self, x, save_dir):  # quest.data_dir
        if save_dir is not None:

            if self.last_save_dir is None or self.last_save_dir == save_dir:
                # normal: push the data and check if need to save
                self.last_save_dir = save_dir
                self.records.append(x)
                self.timestamps.append(time.time())
            
                if len(self.records) > self.chunksize:
                    self.save_and_reset()

            else:
                # suddenly change a dir: save previous chunk first
                self.save_and_reset()

                self.records.append(x)
                self.timestamps.append(time.time())
                self.last_save_dir = save_dir

        else:
            
            if self.last_save_dir is not None:
                # stop saving: save the previous chunk
                self.save_and_reset()

            else:
                # just in case
                self.records = []
                self.timestamps = []

            self.last_save_dir = None
            


    def save_and_reset(self):
        assert self.last_save_dir is not None

        path = f"{self.last_save_dir}/chunk_{self.timestamps[0]}_{self.timestamps[-1]}.npz"
        np.savez(
            path,
            pose=self.records,   # xyz-xyzw
            time=self.timestamps,
        )
        
        self.records = []
        self.timestamps = []
        self.last_save_dir = None

        print(f"Saved chunk to {path}")

            


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="../gloveDemo/leap_assets/leap_hand/robot.urdf",
    )
    # handedness: "right"
    parser.add_argument("--serial_port", type=str, default="COM3")
    parser.add_argument("--serial_baud", type=int, default=115200)
    args = parser.parse_args()

    c = pb.connect(pb.DIRECT)
    vis_sp = []
    c_code = c_code = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
        
    quest = QuestRightArmLeapModule(
        VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=None
    )
    # else:
    #     quest = QuestLeftArmGripperModule(VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=vis_sp)

    start_time = time.time()
    fps_counter = 0
    packet_counter = 0
    print("Initialization completed")
    current_ts = time.time()

    
    data_chunker = DataChunker(chunksize=600)
    last_print = time.time()


    while True:
        now = time.time()
        # TODO: May cause communication issues, need to tune on AR side.
        if now - current_ts < 1 / args.frequency:
            continue
        else:
            current_ts = now

        try:
            wrist, head_pose = quest.receive()

            if wrist:
                data_chunker.put(np.concatenate(wrist), quest.data_dir)

            if time.time() - last_print > 1.0:
                last_print = time.time()
                if wrist is not None:
                    pos, quat = wrist
                    rot = R.from_quat(quat)
                    print("Data:", pos, rot.as_euler("xyz", degrees=True))
                else:
                    print("Data: None")

        except socket.error as e:
            print(e)
            pass

        except KeyboardInterrupt:
            quest.close()
            break

        else:
            packet_time = time.time()
            fps_counter += 1
            packet_counter += 1

            if (packet_time - start_time) > 1.0:
                print(f"received {fps_counter} packets in a second", end="\r")
                start_time += 1.0
                fps_counter = 0
