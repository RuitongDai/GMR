#!/usr/bin/env python3
import numpy as np

def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Quaternion norm is zero.")
    return q / n

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # [w, x, y, z]
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dtype=np.float64)

def axis_angle_to_quat(axis: np.ndarray, theta_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = theta_rad * 0.5
    s = np.sin(h)
    return np.array([np.cos(h), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

def apply_local_xyz_offsets(q_wxyz, dx_deg, dy_deg, dz_deg, order="xyz"):
    q = normalize(np.array(q_wxyz, dtype=np.float64))

    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    angles_deg = {"x": dx_deg, "y": dy_deg, "z": dz_deg}

    # 自身轴旋转：q_new = q ⊗ q_delta（后乘）
    for k in order.lower():
        theta = np.deg2rad(angles_deg[k])
        if abs(theta) < 1e-15:
            continue
        q_delta = axis_angle_to_quat(axes[k], theta)
        q = normalize(quat_mul(q, q_delta))

    return q

if __name__ == "__main__":
    # print("Input quaternion as: w x y z (space-separated)")
    # q_in = list(map(float, input("> ").strip().split()))
    # if len(q_in) != 4:
    #     raise ValueError("Need 4 numbers: w x y z")
    q_in = list([
        -0.5059,
        0.5178,
        0.4929,
        0.4827
            ])
    # q_in = list([0.7071,    0.0,     0.0, 0.7071])
    # q_in = list([1,  0,  0,  0])

    print("Input local angle offsets (degrees) as: dx dy dz (space-separated)")
    dx, dy, dz = map(float, input("> ").strip().split())
    # 如需改变施加顺序，可改 order="zyx" 等
    q_out = apply_local_xyz_offsets(q_in, dx, dy, dz, order="xyz")

    print("Output quaternion [w x y z]:")
    print(f"{q_out[0]:.8f}, {q_out[1]:.8f}, {q_out[2]:.8f}, {q_out[3]:.8f}")
