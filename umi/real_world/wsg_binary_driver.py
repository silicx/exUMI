from typing import Union, Optional
import socket
import enum
import struct

CRC_TABLE_CCITT16 = [
    0x0000,
    0x1021,
    0x2042,
    0x3063,
    0x4084,
    0x50A5,
    0x60C6,
    0x70E7,
    0x8108,
    0x9129,
    0xA14A,
    0xB16B,
    0xC18C,
    0xD1AD,
    0xE1CE,
    0xF1EF,
    0x1231,
    0x0210,
    0x3273,
    0x2252,
    0x52B5,
    0x4294,
    0x72F7,
    0x62D6,
    0x9339,
    0x8318,
    0xB37B,
    0xA35A,
    0xD3BD,
    0xC39C,
    0xF3FF,
    0xE3DE,
    0x2462,
    0x3443,
    0x0420,
    0x1401,
    0x64E6,
    0x74C7,
    0x44A4,
    0x5485,
    0xA56A,
    0xB54B,
    0x8528,
    0x9509,
    0xE5EE,
    0xF5CF,
    0xC5AC,
    0xD58D,
    0x3653,
    0x2672,
    0x1611,
    0x0630,
    0x76D7,
    0x66F6,
    0x5695,
    0x46B4,
    0xB75B,
    0xA77A,
    0x9719,
    0x8738,
    0xF7DF,
    0xE7FE,
    0xD79D,
    0xC7BC,
    0x48C4,
    0x58E5,
    0x6886,
    0x78A7,
    0x0840,
    0x1861,
    0x2802,
    0x3823,
    0xC9CC,
    0xD9ED,
    0xE98E,
    0xF9AF,
    0x8948,
    0x9969,
    0xA90A,
    0xB92B,
    0x5AF5,
    0x4AD4,
    0x7AB7,
    0x6A96,
    0x1A71,
    0x0A50,
    0x3A33,
    0x2A12,
    0xDBFD,
    0xCBDC,
    0xFBBF,
    0xEB9E,
    0x9B79,
    0x8B58,
    0xBB3B,
    0xAB1A,
    0x6CA6,
    0x7C87,
    0x4CE4,
    0x5CC5,
    0x2C22,
    0x3C03,
    0x0C60,
    0x1C41,
    0xEDAE,
    0xFD8F,
    0xCDEC,
    0xDDCD,
    0xAD2A,
    0xBD0B,
    0x8D68,
    0x9D49,
    0x7E97,
    0x6EB6,
    0x5ED5,
    0x4EF4,
    0x3E13,
    0x2E32,
    0x1E51,
    0x0E70,
    0xFF9F,
    0xEFBE,
    0xDFDD,
    0xCFFC,
    0xBF1B,
    0xAF3A,
    0x9F59,
    0x8F78,
    0x9188,
    0x81A9,
    0xB1CA,
    0xA1EB,
    0xD10C,
    0xC12D,
    0xF14E,
    0xE16F,
    0x1080,
    0x00A1,
    0x30C2,
    0x20E3,
    0x5004,
    0x4025,
    0x7046,
    0x6067,
    0x83B9,
    0x9398,
    0xA3FB,
    0xB3DA,
    0xC33D,
    0xD31C,
    0xE37F,
    0xF35E,
    0x02B1,
    0x1290,
    0x22F3,
    0x32D2,
    0x4235,
    0x5214,
    0x6277,
    0x7256,
    0xB5EA,
    0xA5CB,
    0x95A8,
    0x8589,
    0xF56E,
    0xE54F,
    0xD52C,
    0xC50D,
    0x34E2,
    0x24C3,
    0x14A0,
    0x0481,
    0x7466,
    0x6447,
    0x5424,
    0x4405,
    0xA7DB,
    0xB7FA,
    0x8799,
    0x97B8,
    0xE75F,
    0xF77E,
    0xC71D,
    0xD73C,
    0x26D3,
    0x36F2,
    0x0691,
    0x16B0,
    0x6657,
    0x7676,
    0x4615,
    0x5634,
    0xD94C,
    0xC96D,
    0xF90E,
    0xE92F,
    0x99C8,
    0x89E9,
    0xB98A,
    0xA9AB,
    0x5844,
    0x4865,
    0x7806,
    0x6827,
    0x18C0,
    0x08E1,
    0x3882,
    0x28A3,
    0xCB7D,
    0xDB5C,
    0xEB3F,
    0xFB1E,
    0x8BF9,
    0x9BD8,
    0xABBB,
    0xBB9A,
    0x4A75,
    0x5A54,
    0x6A37,
    0x7A16,
    0x0AF1,
    0x1AD0,
    0x2AB3,
    0x3A92,
    0xFD2E,
    0xED0F,
    0xDD6C,
    0xCD4D,
    0xBDAA,
    0xAD8B,
    0x9DE8,
    0x8DC9,
    0x7C26,
    0x6C07,
    0x5C64,
    0x4C45,
    0x3CA2,
    0x2C83,
    0x1CE0,
    0x0CC1,
    0xEF1F,
    0xFF3E,
    0xCF5D,
    0xDF7C,
    0xAF9B,
    0xBFBA,
    0x8FD9,
    0x9FF8,
    0x6E17,
    0x7E36,
    0x4E55,
    0x5E74,
    0x2E93,
    0x3EB2,
    0x0ED1,
    0x1EF0,
]


def checksum_update_crc16(data: bytes, crc: int = 0xFFFF):
    for b in data:
        crc = CRC_TABLE_CCITT16[(crc ^ b) & 0x00FF] ^ (crc >> 8)
    return crc


class StatusCode(enum.IntEnum):
    E_SUCCESS = 0
    E_NOT_AVAILABLE = 1
    E_NO_SENSOR = 2
    E_NOT_INITIALIZED = 3
    E_ALREADY_RUNNING = 4
    E_FEATURE_NOT_SUPPORTED = 5
    E_INCONSISTENT_DATA = 6
    E_TIMEOUT = 7
    E_READ_ERROR = 8
    E_WRITE_ERROR = 9
    E_INSUFFICIENT_RESOURCES = 10
    E_CHECKSUM_ERROR = 11
    E_NO_PARAM_EXPECTED = 12
    E_NOT_ENOUGH_PARAMS = 13
    E_CMD_UNKNOWN = 14
    E_CMD_FORMAT_ERROR = 15
    E_ACCESS_DENIED = 16
    E_ALREADY_OPEN = 17
    E_CMD_FAILED = 18
    E_CMD_ABORTED = 19
    E_INVALID_HANDLE = 20
    E_NOT_FOUND = 21
    E_NOT_OPEN = 22
    E_IO_ERROR = 23
    E_INVALID_PARAMETER = 24
    E_INDEX_OUT_OF_BOUNDS = 25
    E_CMD_PENDING = 26
    E_OVERRUN = 27
    RANGE_ERROR = 28
    E_AXIS_BLOCKED = 29
    E_FILE_EXIST = 30


class CommandId(enum.IntEnum):
    Disconnect = 0x07
    Homing = 0x20
    PrePosition = 0x21
    Stop = 0x22
    FastStop = 0x23
    AckFastStop = 0x24


def args_to_bytes(*args, int_bytes=1):
    buf = list()
    for arg in args:
        if isinstance(arg, float):
            # little endian 32bit float
            buf.append(struct.pack("<f", arg))
        elif isinstance(arg, int):
            buf.append(arg.to_bytes(length=int_bytes, byteorder="little"))
        elif isinstance(arg, str):
            buf.append(arg.encode("ascii"))
        else:
            raise RuntimeError(f"Unsupported type {type(arg)}")
    result = b"".join(buf)
    return result


class WSGBinaryDriver:
    def __init__(self, hostname="192.168.0.103", port=1000):
        self.hostname = hostname
        self.port = port
        self.tcp_sock = None

    def start(self):
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((self.hostname, self.port))
        # self.ack_fast_stop()

    def stop(self):
        self.stop_cmd()
        self.disconnect()
        self.tcp_sock.close()
        return

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ================= low level API ================

    def msg_send(self, cmd_id: int, payload: bytes):
        preamble_b = 0xAA.to_bytes(1, "little") * 3
        cmd_b = int(cmd_id).to_bytes(1, "little")
        size_b = len(payload).to_bytes(2, "little")
        msg_b = preamble_b + cmd_b + size_b + payload
        checksum_b = checksum_update_crc16(msg_b).to_bytes(2, "little")
        msg_b += checksum_b
        return self.tcp_sock.send(msg_b)

    def msg_receive(self) -> dict:
        # syncing
        sync = 0
        while sync != 3:
            res = self.tcp_sock.recv(1)
            if res == 0xAA.to_bytes(1, "little"):
                sync += 1

        # read header
        cmd_id_b = self.tcp_sock.recv(1)
        cmd_id = int.from_bytes(cmd_id_b, "little")

        # read size
        size_b = self.tcp_sock.recv(2)
        size = int.from_bytes(size_b, "little")

        # read payload
        payload_b = self.tcp_sock.recv(size)
        status_code = int.from_bytes(payload_b[:2], "little")

        parameters_b = payload_b[2:]

        # read checksum
        checksum_b = self.tcp_sock.recv(2)

        # correct checksum ends in zero
        header_checksum = 0x50F5
        msg_checksum = checksum_update_crc16(
            cmd_id_b + size_b + payload_b + checksum_b, crc=header_checksum
        )
        if msg_checksum != 0:
            raise RuntimeError("Corrupted packet received from WSG")

        result = {
            "command_id": cmd_id,
            "status_code": status_code,
            "payload_bytes": parameters_b,
        }
        return result

    def cmd_submit(
        self,
        cmd_id: int,
        payload: bytes = b"",
        pending: bool = True,
        ignore_other=False,
    ):
        res = self.msg_send(cmd_id, payload)
        if res < 0:
            raise RuntimeError("Message send failed.")

        # receive response, repeat if pending
        msg = None
        keep_running = True
        while keep_running:
            msg = self.msg_receive()
            if ignore_other and msg["command_id"] != cmd_id:
                continue

            if msg["command_id"] != cmd_id:
                raise RuntimeError(
                    "Response ID ({:02X}) does not match submitted command ID ({:02X})\n".format(
                        msg["command_id"], cmd_id
                    )
                )
            if pending:
                status = msg["status_code"]
            keep_running = pending and status == StatusCode.E_CMD_PENDING.value
        return msg

    # ============== mid level API ================

    def act(self, cmd: CommandId, *args, wait=True, ignore_other=False):
        msg = self.cmd_submit(
            cmd_id=cmd.value,
            payload=args_to_bytes(*args),
            pending=wait,
            ignore_other=ignore_other,
        )
        msg["command_id"] = CommandId(msg["command_id"])
        msg["status_code"] = StatusCode(msg["status_code"])

        status = msg["status_code"]
        if status != StatusCode.E_SUCCESS:
            raise RuntimeError(f"Command {cmd} not successful: {status}")
        return msg

    # =============== high level API ===============

    def disconnect(self):
        # use msg_send to no wait for response
        return self.msg_send(CommandId.Disconnect.value, b"")

    def homing(self, positive_direction=True, wait=True):
        arg = 0
        if positive_direction is None:
            arg = 0
        elif positive_direction:
            arg = 1
        else:
            arg = 2

        return self.act(CommandId.Homing, arg, wait=wait)

    def pre_position(
        self, width: float, speed: float, clamp_on_block: bool = True, wait=True
    ):
        flag = 0
        if clamp_on_block:
            flag = 0
        else:
            flag = 1

        return self.act(
            CommandId.PrePosition, flag, float(width), float(speed), wait=wait
        )

    def ack_fault(self):
        return self.act(CommandId.AckFastStop, "ack", wait=False, ignore_other=True)

    def stop_cmd(self):
        return self.act(CommandId.Stop, wait=False, ignore_other=True)

    def custom_script(self, cmd_id: int, *args):
        # Custom payload format:
        # 0:	Unused
        # 1..4	float
        # .... one float each
        payload_args = [0]
        for arg in args:
            payload_args.append(float(arg))
        payload = args_to_bytes(*payload_args, int_bytes=1)

        # send message
        msg = self.cmd_submit(cmd_id=cmd_id, payload=payload, pending=False)
        status = StatusCode(msg["status_code"])
        response_payload = msg["payload_bytes"]
        if status == StatusCode.E_CMD_UNKNOWN:
            raise RuntimeError(
                "Command unknown - make sure script (cmd_measure.lua) is running"
            )
        if status != StatusCode.E_SUCCESS:
            raise RuntimeError("Command failed")
        if len(response_payload) != 17:
            raise RuntimeError(
                "Response payload incorrect (",
                "".join("{:02X}".format(b) for b in response_payload),
                ")",
            )

        # parse payload
        state = response_payload[0]
        values = list()
        for i in range(4):
            start = i * 4 + 1
            end = start + 4
            values.append(struct.unpack("<f", response_payload[start:end])[0])

        info = {
            "state": state,
            "position": values[0],
            "velocity": values[1],
            "force_motor": values[2],
            "measure_timestamp": values[3],
            "is_moving": (state & 0x02) != 0,
        }
        # info = {
        #     'state': 0,
        #     'position': 100.,
        #     'velocity': 0.,
        #     'force_motor': 0.,
        #     'is_moving': 0.
        # }
        return info

    def script_query(self):
        return self.custom_script(0xB0)

    def script_position_pd(
        self,
        position: float,
        velocity: float,
        kp: float = 15.0,
        kd: float = 1e-3,
        travel_force_limit: float = 80.0,
        blocked_force_limit: float = None,
    ):
        if blocked_force_limit is None:
            blocked_force_limit = travel_force_limit
        assert kp > 0
        assert kd >= 0
        return self.custom_script(
            0xB1, position, velocity, kp, kd, travel_force_limit, blocked_force_limit
        )


def test():
    import numpy as np
    import time

    with WSGBinaryDriver(
        hostname="wsg50-00004544.internal.tri.global", port=1000
    ) as wsg:
        # ACK
        # msg = wsg.cmd_submit(0x24, bytearray([0x61, 0x63, 0x6B]))
        msg = wsg.ack_fault()
        print(msg)

        # HOME
        # msg = wsg.cmd_submit(0x20, bytearray([0x01]))
        msg = wsg.homing()
        print(msg)
        # time.sleep(1.0)

        # msg = wsg.pre_position(0, 150)
        # print(msg)
        # time.sleep(1.0)

        T = 2
        dt = 1 / 30
        pos = np.linspace(0.0, 110.0, int(T / dt))[::-1]
        vel = np.diff(pos) / dt
        vel = np.append(vel, vel[-1])

        t_start = time.time()
        for i in range(len(pos)):
            p = pos[i]
            v = vel[i]
            print(p, v)
            info = wsg.script_position(position=p, dt=dt)
            print(info)

            t_end = t_start + i * dt
            t_sleep = t_end - time.time()
            print(t_sleep)
            if t_sleep > 0:
                time.sleep(t_sleep)
        print(time.time() - t_start)
        # cmd_id_b, payload_b, checksum_b = wsg.msg_receive()
        # cmd_id_b, payload_b, checksum_b = wsg.msg_receive()
        time.sleep(3.0)

        T = 2
        dt = 1 / 30
        pos = np.linspace(0.0, 110.0, int(T / dt))
        vel = np.diff(pos) / dt
        vel = np.append(vel, vel[-1])

        t_start = time.time()
        for i in range(len(pos)):
            p = pos[i]
            v = vel[i]
            print(p, v)
            info = wsg.script_position(position=p, dt=dt)
            print(info)

            t_end = t_start + i * dt
            t_sleep = t_end - time.time()
            print(t_sleep)
            if t_sleep > 0:
                time.sleep(t_sleep)
        print(time.time() - t_start)

        # wsg.msg_send(0x30, bytearray([0x00, 0x00, 0x00, 0x00, 0x16, 0x43]))
        # cmd_id_b, payload_b, checksum_b = wsg.msg_receive()
        # time.sleep(1.0)
