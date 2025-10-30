import numpy as np
from scipy.spatial.transform import Rotation as R


__all__ = [
    "ARUCO_ID",
    "tx_arhand_inv",
    "tx_arbase_at_flexivbase",
    "tx_flexivobj_at_arobj",
    "tx_flexivcamera_at_flexivobj",
]


ARUCO_ID = 4   # aruco id to be used for alignment



tx_arhand = np.identity(4)  # pose of arhand at arbase coordinates
tx_arhand[:3, 3] = [0.03227652, -0.0598839, -0.0737011]
tx_arhand[:3, :3] = R.from_euler('xyz', [16.2832578, -1.30209369, 23.76593208], degrees=True).as_matrix()
tx_arhand_inv = np.linalg.inv(tx_arhand)


# tx_flexivbase = np.identity(4)  # pose of flexiv-base at arbase coordinates
# tx_flexivbase[:3, :3] =  R.from_euler('z', [90], degrees=True).as_matrix()
# tx_flexivbase_inv = np.linalg.inv(tx_flexivbase)

# tx_rotate_camera = np.identity(4)
# tx_rotate_camera[:3, :3] = R.from_euler('y', [90], degrees=True).as_matrix()


tx_arbase_at_flexivbase = np.identity(4)  # pose of flexiv-base at arbase coordinates
tx_arbase_at_flexivbase[:3, :3] = R.from_euler("z", [-90], degrees=True).as_matrix()

tx_flexivobj_at_arobj = np.identity(4)  # pose of flexiv-base at arbase coordinates
tx_flexivobj_at_arobj[:3, :3] = R.from_euler("z", [90], degrees=True).as_matrix()

tx_flexivcamera_at_flexivobj = np.identity(4)
tx_flexivcamera_at_flexivobj[:3, :3] = R.from_euler("y", [90], degrees=True).as_matrix()
