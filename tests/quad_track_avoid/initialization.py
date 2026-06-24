import json
import numpy as np

with open("tests/quad_track_avoid/test_params.json") as f:
    tmp = json.load(f)
    obstacles = tmp["obstacles"]
    # Pringle reference parameters
    PRINGLE_A = tmp["pringle_a"]
    PRINGLE_B = tmp["pringle_b"]
    PRINGLE_C = tmp["pringle_c"]


def obstacle_centers_at_t0(obstacles_=obstacles):
    """Obstacle (xc, yc, r) at simulation t=0 (used for initial-condition rejection)."""
    out = []
    for obs in obstacles_:
        xc = obs["xc0"] + obs["ax"] * np.sin(obs["px"])
        yc = obs["yc0"] + obs["ay"] * np.sin(obs["py"])
        out.append((xc, yc, obs["r"]))
    return out

def YPRToQuat_np(r1, r2, r3):
    cr1, cr2, cr3 = np.cos(0.5*r1), np.cos(0.5*r2), np.cos(0.5*r3)
    sr1, sr2, sr3 = np.sin(0.5*r1), np.sin(0.5*r2), np.sin(0.5*r3)
    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3
    q = np.array([q0, q1, q2, q3])
    return q / np.linalg.norm(q)

def pringle_state_np(s, avg_vel, A=PRINGLE_A, B=PRINGLE_B, C=PRINGLE_C):
    """Full 13-state pringle reference at parameter s (numpy)."""
    x_ = A * np.cos(s)
    y_ = B * np.sin(s)
    z_ = C * np.cos(2*s)
    xdot = -A * np.sin(s) * avg_vel
    ydot =  B * np.cos(s) * avg_vel
    zdot = -2*C * np.sin(2*s) * avg_vel
    yaw = np.arctan2(ydot, xdot)
    q0, q1, q2, q3 = YPRToQuat_np(yaw, 0, 0)
    return np.array([x_, y_, z_, q0, q1, q2, q3, xdot, ydot, zdot, 0, 0, 0])

def build_xr_from_s(s_start, N, Ts, avg_vel, A=PRINGLE_A, B=PRINGLE_B, C=PRINGLE_C):
    """Reference horizon (N, 13) starting at pringle parameter s_start."""
    xr_arr = np.zeros((N, 13), dtype=np.float64)
    for k in range(N):
        sk = s_start + k * Ts * avg_vel
        xr_arr[k] = pringle_state_np(sk, avg_vel, A, B, C)
    return xr_arr

def generate_random_inits(N_runs, N_horizon, Ts, avg_vel, delta_std=0.5,
                          margin=0.1, seed=0, max_attempts_per=200):
    """Generate N_runs random (x0, xr, s) triples with obstacle rejection.

    - s ~ uniform[0, 2π]
    - x0 = pringle_state(s) + (δ_xy in 2D normal × delta_std, δ_z too); quaternion=identity, vel=0
    - reject x0 whose xy lies within `margin` m of any obstacle center (radius+margin).
    """
    rng = np.random.default_rng(seed)
    obs_t0 = obstacle_centers_at_t0()
    x0_list = np.zeros((N_runs, 13), dtype=np.float64)
    xr_list = np.zeros((N_runs, N_horizon, 13), dtype=np.float64)
    s_list = np.zeros(N_runs, dtype=np.float64)
    rejects = 0
    for i in range(N_runs):
        for _ in range(max_attempts_per):
            s = float(rng.uniform(0.0, 2.0 * np.pi))
            ref0 = pringle_state_np(s, avg_vel)
            delta = rng.standard_normal(3) * delta_std
            x0 = np.zeros(13, dtype=np.float64)
            x0[:3] = ref0[:3] + delta
            x0[3] = 1.0  # identity quaternion (rest 0)
            ok = True
            for xc, yc, r in obs_t0:
                if (x0[0] - xc) ** 2 + (x0[1] - yc) ** 2 < (r + margin) ** 2:
                    ok = False
                    rejects += 1
                    break
            if ok:
                break
        else:
            # max_attempts hit — fall back to the reference itself (no offset).
            # Pringle origin is far from obstacles by design, so this is safe.
            s = float(rng.uniform(0.0, 2.0 * np.pi))
            ref0 = pringle_state_np(s, avg_vel)
            x0 = np.zeros(13, dtype=np.float64)
            x0[:3] = ref0[:3]
            x0[3] = 1.0
        s_list[i] = s
        x0_list[i] = x0
        xr_list[i] = build_xr_from_s(s, N_horizon, Ts, avg_vel)
    return x0_list, xr_list, s_list, rejects

