import os
import time
import enum
import sys
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
import json

from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from umi.common.pose_util import *


from .flexiv import *

sys.path.append(os.path.join(os.getcwd(), "flexiv_api/lib_py/"))

try:
    import flexivrdk
except ModuleNotFoundError:
    import warnings
    warnings.warn("No module named 'flexivrdk', but continue anyway")

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESTART_PUT = 3



class GripperWidthMapper:
    def __init__(self, calibration_path="example/calibration/real_to_aruco_width.json"):
        calib_data = json.load(open(calibration_path, "r"))

        calib_data = self.__merge_similar_points(calib_data)
        self.__assert_monotonous(calib_data)
        print("calib_data", calib_data)

        real_width, aruco_width = [x for x, _ in calib_data], [y for _, y in calib_data]
        self.interp_r2a = si.interp1d(
            real_width,
            aruco_width,
            bounds_error=False,
            fill_value=(np.min(aruco_width), np.max(aruco_width)),
        )
        self.interp_a2r = si.interp1d(
            aruco_width,
            real_width,
            bounds_error=False,
            fill_value=(np.min(real_width), np.max(real_width)),
        )

    def __assert_monotonous(self, data):
        for i in range(1, len(data)):
            assert data[i] >= data[i - 1]

    def __merge_similar_points(self, data, threshold=1.0e-4):
        data = sorted(data, key=lambda x: x[0])

        new_data = []
        last_r, last_a = [], []
        for r, a in data:
            if len(last_r) == 0 or np.abs(r - np.mean(last_r)) < threshold:
                last_r.append(r)
                last_a.append(a)
            else:
                new_data.append([np.mean(last_r), np.mean(last_a)])
                last_r, last_a = [], []

        if len(last_r) > 0:
            new_data.append([np.mean(last_r), np.mean(last_a)])

        return new_data

    def real_to_aruco(self, real_width):
        width = self.interp_r2a(real_width)
        return width

    def aruco_to_real(self, aruco_width):
        width = self.interp_a2r(aruco_width)
        return width
    

class NullWidthMapper:
    def __init__(self):
        pass

    def real_to_aruco(self, real_width):
        return real_width

    def aruco_to_real(self, aruco_width):
        return aruco_width
    


class FlexivInterface:
    DOF = 7
    TARGET_VEL = [0.0] * DOF
    TARGET_ACC = [0.0] * DOF
    MAX_VEL = [ float(os.environ.get("FLEXIV_MAX_VEL", "0.1")) ] * DOF
    MAX_ACC = [ float(os.environ.get("FLEXIV_MAX_ACC", "0.2")) ] * DOF


    tx_flange_tip = np.identity(4)
    # tx_flange_tip[:3, 3] = np.array([-0.04, 0, 0.19])  # calibrated
    tx_flange_tip[:3, 3] = np.array([0, 0, 0.185])  # measured physically
    tx_tip_flange = np.linalg.inv(tx_flange_tip)


    @staticmethod
    def tip_to_flange_pose(tip_pose):
        return mat_to_pose(pose_to_mat(tip_pose) @ FlexivInterface.tx_tip_flange)


    def __init__(self, robot_ip, local_ip, move_home=True, init_offset=None, init_qpos=None, use_gripper_width_mapping=True, device="cuda"):
        self.log = log = flexivrdk.Log()
        self.mode = flexivrdk.Mode

        self.sim_rizon = sim_rizon = Rizon4(headless=True)
        self.robot = robot = flexivrdk.Robot(robot_ip, local_ip)
        self.robot_states = robot_states = flexivrdk.RobotStates()
        self.gripper = flexivrdk.Gripper(robot)
        self.gripper_states = flexivrdk.GripperStates()

        if use_gripper_width_mapping:
            self.gripper_width_mapper = GripperWidthMapper()
        else:
            self.gripper_width_mapper = NullWidthMapper()

        self.verbose = os.environ.get("FLEXIV_VERBOSE", "0") == "1"

        flange_lower_limits, flange_upper_limits = sim_rizon.get_flange_limits()
        flange_lower_limits = torch.from_numpy(flange_lower_limits).to(device)
        flange_upper_limits = torch.from_numpy(flange_upper_limits).to(device)

        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        log.info("Enabling robot ...")
        robot.enable()
        while not robot.isOperational():
            time.sleep(1)

        self.last_send_pose = None

        # =============== init pose =========================

        assert not (init_offset is not None and not move_home), "init_offset is only valid when move_home is True"

        self.robot.setMode(self.mode.NRT_JOINT_POSITION)
        self.robot.getRobotStates(self.robot_states)
        self.gripper.move(0.12, 0.1, 5)
        if move_home:
            self.move_to_home()

        self.robot.setMode(self.mode.NRT_JOINT_POSITION)

        if init_qpos is not None:
            assert len(init_qpos) == 7
            self.send_joint_position(np.array(init_qpos))
            time.sleep(10)
            print("Set to desired q pos: ", init_qpos)

        if init_offset is not None:
            assert len(init_offset) == 3
            pose = self.get_ee_pose()
            pose = FlexivInterface.tip_to_flange_pose(pose)
            pos, rot = pose_to_pos_rot(pose)
            pos[0] += init_offset[0]
            pos[1] += init_offset[1]
            pos[2] += init_offset[2]
            pose = pos_rot_to_pose(pos, rot)
            self.send_flange_pose(pose)
            print("Set position offset: ", init_offset)
            time.sleep(12)

        self.log.info("Done robot initializing")

        self.robot.getRobotStates(self.robot_states)
        self.sim_rizon.set_joints(self.robot_states.q)


    def move_to_home(self):
        self.log.info("Move to home")
        # robot
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.executePrimitive("Home()")
        while self.robot.isBusy():
            time.sleep(1)
        self.robot.executePrimitive("ZeroFTSensor()")

        # gripper
        self.gripper.move(0.12, 0.1, 5)

        time.sleep(1.0)
        self.log.info("Moved home")
        print(self.get_gripper_width())

    # robot arm api

    def get_flange_pose(self):
        """return pose in flexiv's coordinates"""
        self.robot.getRobotStates(self.robot_states)

        # ## ABANDON: direct read flangePose from api
        flange_pose = self.robot_states.flangePose  # pppqqqq wzyx quat for flexiv api
        pos, quat = flange_pose[:3], flange_pose[3:]
        pos = np.array(pos)
        rot = st.Rotation.from_quat(quat, scalar_first=True)

        ### read flangepose from pybullet
        # self.sim_rizon.set_joints(self.get_joint_positions())
        # flange_pose = self.sim_rizon.get_catersian(self.sim_rizon.flange_link)
        # pos, quat = flange_pose[:3], flange_pose[3:]  # xyzw quat for pybullet
        # rot = st.Rotation.from_quat(quat, scalar_first=False)

        return pos_rot_to_pose(pos, rot)

    def get_ee_pose(self):
        
        flange_pose = self.get_flange_pose()
        pos, rot = pose_to_pos_rot(flange_pose)

        tip_pose_mat = pos_rot_to_mat(np.array(pos), rot) @ FlexivInterface.tx_flange_tip
        umi_tip_pose = mat_to_pose(tip_pose_mat)

        return umi_tip_pose

    def get_joint_positions(self):
        self.robot.getRobotStates(self.robot_states)
        return np.array(self.robot_states.q)

    def get_joint_velocities(self):
        self.robot.getRobotStates(self.robot_states)
        return np.array(self.robot_states.dq)

    def send_flange_pose(self, flange_pose: np.ndarray):
        # print("Arm: ", time.monotonic(), flange_pose)
        """receive pose in flexiv's coordinates, not umi coordinates"""
        self.sim_rizon.set_joints(self.get_joint_positions())

        ## Protect
        tcp_pose = mat_to_pose(pose_to_mat(flange_pose) @ FlexivInterface.tx_flange_tip)
        tcp_pose[2] = max(tcp_pose[2], 0.02)  # limit z
        flange_pose = mat_to_pose(pose_to_mat(tcp_pose) @ FlexivInterface.tx_tip_flange)


        # from pos-rotvec 6d pose to pos-quat 7d pose
        pos, rot = pose_to_pos_rot(flange_pose)
        quat = rot.as_quat(scalar_first=False)  # zyxw quat for scipy/pybullet api
        flange_pose = np.concatenate([pos, quat])
        next_joints = self.sim_rizon.calc_ik(flange_pose)  # pppqqqq -> q

        if self.verbose:
            print(
                "[FlexivInterface] [DEBUG] Sending flange pose:",
                " ".join(["%4.2f" % x for x in flange_pose]),
                "(q =",
                " ".join(["%4.2f" % x for x in next_joints]),
                ")",
            )

        self.send_joint_position(next_joints)

    def send_joint_position(self, positions: np.ndarray):
        
        if os.environ.get("FLEXIV_USE_VEL_CONTROL", "0") == "1":
            if self.last_send_pose is None:
                target_vel = np.zeros_like(positions)
            else:
                curr_pos = self.get_joint_positions()
                curr_t = time.time()
                target_vel = (positions - curr_pos) / (curr_t - self.last_send_pose)
                target_vel[target_vel > FlexivInterface.MAX_VEL] = FlexivInterface.MAX_VEL
                self.last_send_pose = curr_t
        else:
            target_vel = FlexivInterface.TARGET_VEL

        # for rizon4
        self.robot.sendJointPosition(
            positions,
            target_vel,
            FlexivInterface.TARGET_ACC,
            FlexivInterface.MAX_VEL,
            FlexivInterface.MAX_ACC,
        )

    # gripper api

    def send_gripper_state(self, pos: float, vel: float, force: float):
        # print("Gripper: ", time.monotonic(), pos)
        target_pos = self.gripper_width_mapper.aruco_to_real(pos)

        print("Send gripper", "%5.2f"%target_pos) 

        if self.verbose:
            print("[FlexivInterface] [DEBUG] Gripper move to %.5f" % target_pos)
        self.gripper.move(target_pos, vel, force)

    def get_gripper_width(self):
        self.gripper.getGripperStates(self.gripper_states)

        aruco_width = self.gripper_width_mapper.real_to_aruco(self.gripper_states.width)
        return aruco_width

    def get_gripper_force(self):
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.force

    def get_gripper_state(self):
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.width, self.gripper_states.force


class FlexivInterpolationController(mp.Process):
    """ """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        local_ip,
        frequency=1000,
        launch_timeout=3,
        joints_init=None,
        soft_real_time=False,
        move_max_speed=0.1,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        command_queue_size=1024,  # for gripper
    ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FlexivController")
        self.robot_ip = robot_ip
        self.local_ip = local_ip

        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.move_max_speed = move_max_speed

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "duration": 0.0,
            "target_time": 0.0,
        }
        robot_input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )

        example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": 0.0,
            "target_time": 0.0,
        }
        gripper_input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=command_queue_size
        )

        # build ring buffer
        robot_receive_keys = [
            ("ActualTCPPose", "get_ee_pose"),
            ("ActualQ", "get_joint_positions"),
            ("ActualQd", "get_joint_velocities"),
        ]
        example = dict()
        for key, func_name in robot_receive_keys:
            if "joint" in func_name:
                example[key] = np.zeros(7)
            elif "ee_pose" in func_name:
                example[key] = np.zeros(6)

        example["robot_receive_timestamp"] = time.time()
        example["robot_timestamp"] = time.time()
        robot_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        example = {
            "gripper_position": 0.0,
            "gripper_force": 0.0,
            "gripper_receive_timestamp": time.time(),
            "gripper_timestamp": time.time(),
        }
        gripper_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.robot_input_queue = robot_input_queue
        self.gripper_input_queue = gripper_input_queue
        self.robot_ring_buffer = robot_ring_buffer
        self.gripper_ring_buffer = gripper_ring_buffer

        self.robot_receive_keys = robot_receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(
                f"[FlexivPositionalController] Controller process spawned at {self.pid}"
            )

    def stop(self, wait=True):
        self.robot_input_queue.put({"cmd": Command.STOP.value})

        self.gripper_input_queue.put({"cmd": Command.STOP.value})

        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def robot_servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "duration": duration,
        }
        self.robot_input_queue.put(message)

    def robot_schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "target_time": target_time,
        }
        self.robot_input_queue.put(message)

    def gripper_schedule_waypoint(self, pos: float, target_time: float):
        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": pos,
            "target_time": target_time,
        }
        self.gripper_input_queue.put(message)

    def gripper_restart_put(self, start_time):
        self.gripper_input_queue.put(
            {"cmd": Command.RESTART_PUT.value, "target_time": start_time}
        )

    # ========= receive APIs =============
    def get_robot_state(self, k=None, out=None):
        if k is None:
            return self.robot_ring_buffer.get(out=out)
        else:
            return self.robot_ring_buffer.get_last_k(k=k, out=out)

    def get_robot_all_state(self):
        return self.robot_ring_buffer.get_all()

    def get_gripper_state(self, k=None, out=None):
        if k is None:
            return self.gripper_ring_buffer.get(out=out)
        else:
            return self.gripper_ring_buffer.get_last_k(k=k, out=out)

    def get_gripper_all_state(self):
        return self.gripper_ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        # start flexiv rizon4 interface
        robot = FlexivInterface(
            robot_ip=self.robot_ip, local_ip=self.local_ip, device="cuda",
            move_home=False, 
            init_qpos=eval(os.environ.get("FLEXIV_INIT_POSE", "Set the pose in environ")),
            use_gripper_width_mapping=False
        )
        
        if self.verbose:
            print(f"[FlexivPositionalController] Connected to robot")

        # init robot pose
        if self.joints_init is not None:
            assert len(self.joints_init) == 7
            robot.send_joint_position(self.joints_init)

        try:
            # main loop
            dt = 1.0 / self.frequency

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            ### robot
            curr_pose = robot.get_ee_pose()
            robot_pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t], poses=[curr_pose]
            )
            ### gripper
            curr_pos = robot.get_gripper_width()
            gripper_pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t], poses=[[curr_pos, 0, 0, 0, 0, 0]]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:

                #! Send command to robot
                t_now = time.monotonic()
                print("t_now", t_now)
                # diff = t_now - robot_pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = robot_pose_interp(t_now)
                # print("Received    (tip)", tip_pose)
                flange_pose = FlexivInterface.tip_to_flange_pose(tip_pose)
                robot.send_flange_pose(flange_pose)

                # print("robot_pose_interp: ")
                # pprint_pose = lambda p: ", ".join(["%5.3f"%x for x in p])
                # ct = time.monotonic()
                # for t, p in zip(robot_pose_interp.times, robot_pose_interp.poses):
                #     print("t: %5.2f"%(t-ct), "pose: ", pprint_pose(p))
                # print("send tip pose:", pprint_pose(tip_pose))
                # print("send flange pose:", pprint_pose(flange_pose))
                # print("==================================================")

                #! send command to gripper
                # t_now = time.monotonic()
                target_pos = gripper_pose_interp(t_now)[0]
                # target_vel = (target_pos - gripper_pose_interp(t_now - dt)[0]) / dt
                target_pos = max(target_pos, 0.005)
                robot.send_gripper_state(target_pos, 0.1, 10)

                # update robot state
                state = dict()
                for key, func_name in self.robot_receive_keys:
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state["robot_receive_timestamp"] = t_recv
                state["robot_timestamp"] = t_recv - self.receive_latency
                self.robot_ring_buffer.put(state)

                # update gripper state
                width, force = robot.get_gripper_state()
                state = {
                    "gripper_position": width,
                    "gripper_force": force,
                    "gripper_receive_timestamp": time.time(),
                    "gripper_timestamp": time.time() - self.receive_latency,
                }
                self.gripper_ring_buffer.put(state)

                # fetch command from robot queue
                try:
                    commands = self.robot_input_queue.get_all()
                    # commands = self.robot_input_queue.get_k(1)  # process at most 1 command per cycle to maintain frequency
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute robot commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        robot_pose_interp = robot_pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                "[FlexivPositionalController] New pose target:{} duration:{}s".format(
                                    target_pose, duration
                                )
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        robot_pose_interp = robot_pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # fetch command from gripper queue
                try:
                    commands = self.gripper_input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command["target_pos"]
                        target_time = command["target_time"]
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        gripper_pose_interp = gripper_pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = (
                            command["target_time"] - time.time() + time.monotonic()
                        )
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(
                        f"[FlexivPositionalController] Actual frequency {1/(time.monotonic() - t_now)}"
                    )

        except:
            import traceback

            traceback.print_exc()
            raise ValueError()

        finally:
            # manditory cleanup
            # terminate
            # print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            # robot.terminate_current_policy()
            # del robot
            self.ready_event.set()

            if self.verbose:
                print(
                    f"[FlexivPositionalController] Disconnected from robot: {self.robot_ip}"
                )
