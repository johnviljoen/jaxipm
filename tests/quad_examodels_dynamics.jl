function quad_xdot(xi, uj, k, qp; g = 9.81)
    mB, Cd, kTh, kTo = qp.mB, qp.Cd, qp.kTh, qp.kTo
    dxm, dym, IRzz   = qp.dxm, qp.dym, qp.IRzz
    useP             = qp.usePrecession ? 1.0 : 0.0
    Ixx, Iyy, Izz    = qp.IB00, qp.IB11, qp.IB22   # diagonal inertia (Ixx = Iyy here)

    # Total thrust coefficient Σ kTh·wᵢ²  (the `th` sum in the Python form).
    thsum = kTh * (uj(k,1)^2 + uj(k,2)^2 + uj(k,3)^2 + uj(k,4)^2)

    # Positions: ẋ = xd, ẏ = yd, ż = zd
    f1 = xi(k, 8)
    f2 = xi(k, 9)
    f3 = xi(k, 10)

    # Quaternion kinematics
    f4 = -0.5*xi(k,11)*xi(k,5) - 0.5*xi(k,12)*xi(k,6) - 0.5*xi(k,7)*xi(k,13)
    f5 =  0.5*xi(k,11)*xi(k,4) - 0.5*xi(k,12)*xi(k,7) + 0.5*xi(k,6)*xi(k,13)
    f6 =  0.5*xi(k,11)*xi(k,7) + 0.5*xi(k,12)*xi(k,4) - 0.5*xi(k,5)*xi(k,13)
    f7 = -0.5*xi(k,11)*xi(k,6) + 0.5*xi(k,12)*xi(k,5) + 0.5*xi(k,4)*xi(k,13)

    # Translational accelerations (drag = -Cd·v·abs(v); thrust = kTh·Σw²)
    f8  = (-Cd * xi(k,8) * abs(xi(k,8))
           - 2.0*(xi(k,4)*xi(k,6) + xi(k,5)*xi(k,7)) * thsum) / mB
    f9  = (-Cd * xi(k,9) * abs(xi(k,9))
           + 2.0*(xi(k,4)*xi(k,5) - xi(k,6)*xi(k,7)) * thsum) / mB
    f10 = (-Cd * xi(k,10) * abs(xi(k,10))
           - thsum * (xi(k,4)^2 - xi(k,5)^2 - xi(k,6)^2 + xi(k,7)^2)
           + g * mB) / mB

    # Angular accelerations
    f11 = ((Iyy - Izz) * xi(k,12) * xi(k,13)
           - useP * IRzz * (uj(k,1) - uj(k,2) + uj(k,3) - uj(k,4)) * xi(k,12)
           + kTh * (uj(k,1)^2 - uj(k,2)^2 - uj(k,3)^2 + uj(k,4)^2) * dym) / Ixx
    f12 = ((Izz - Ixx) * xi(k,11) * xi(k,13)
           + useP * IRzz * (uj(k,1) - uj(k,2) + uj(k,3) - uj(k,4)) * xi(k,11)
           + kTh * (uj(k,1)^2 + uj(k,2)^2 - uj(k,3)^2 - uj(k,4)^2) * dxm) / Iyy
    f13 = ((Ixx - Iyy) * xi(k,11) * xi(k,12)
           + kTo * (-uj(k,1)^2 + uj(k,2)^2 - uj(k,3)^2 + uj(k,4)^2)) / Izz

    return (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13)
end
