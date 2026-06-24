import casadi as ca

def f_ca(x, u, qp, d=[0.]*3, g=9.81):
    """CasADi SX quadcopter rigid-body dynamics. Byte-for-byte copy of
    gauntlet's f_ca."""
    th = qp['kTh'] * u ** 2
    to = qp['kTo'] * u ** 2

    xd = ca.vertcat(
        x[7], x[8], x[9],
        - 0.5 * x[10] * x[4] - 0.5 * x[11] * x[5] - 0.5 * x[6] * x[12],
          0.5 * x[10] * x[3] - 0.5 * x[11] * x[6] + 0.5 * x[5] * x[12],
          0.5 * x[10] * x[6] + 0.5 * x[11] * x[3] - 0.5 * x[4] * x[12],
        - 0.5 * x[10] * x[5] + 0.5 * x[11] * x[4] + 0.5 * x[3] * x[12],
        (
            qp["Cd"]
            * ca.sign(d[0] * ca.cos(d[1]) * ca.cos(d[2]) - x[7])
            * (d[0] * ca.cos(d[1]) * ca.cos(d[2]) - x[7]) ** 2
            - 2 * (x[3] * x[5] + x[4] * x[6]) * (th[0] + th[1] + th[2] + th[3])
        )
        / qp["mB"],
        (
            qp["Cd"]
            * ca.sign(d[0] * ca.sin(d[1]) * ca.cos(d[2]) - x[8])
            * (d[0] * ca.sin(d[1]) * ca.cos(d[2]) - x[8]) ** 2
            + 2 * (x[3] * x[4] - x[5] * x[6]) * (th[0] + th[1] + th[2] + th[3])
        )
        / qp["mB"],
        (
            -qp["Cd"] * ca.sign(d[0] * ca.sin(d[2]) + x[9]) * (d[0] * ca.sin(d[2]) + x[9]) ** 2
            - (th[0] + th[1] + th[2] + th[3])
            * (x[3] ** 2 - x[4] ** 2 - x[5] ** 2 + x[6] ** 2)
            + g * qp["mB"]
        )
        / qp["mB"],
        (
            (qp["IB"][1,1] - qp["IB"][2,2]) * x[11] * x[12]
            - qp["usePrecession"] * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3]) * x[11]
            + (th[0] - th[1] - th[2] + th[3]) * qp["dym"]
        )
        / qp["IB"][0,0],
        (
            (qp["IB"][2,2] - qp["IB"][0,0]) * x[10] * x[12]
            + qp["usePrecession"] * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3]) * x[10]
            + (th[0] + th[1] - th[2] - th[3]) * qp["dxm"]
        )
        / qp["IB"][1,1],
        ((qp["IB"][0,0] - qp["IB"][1,1]) * x[10] * x[11] - to[0] + to[1] - to[2] + to[3]) / qp["IB"][2,2],
    )

    return xd