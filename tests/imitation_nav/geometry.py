import jax.numpy as jnp

# def quaternion_to_dcm(q):
#     dcm = np.zeros([3,3])

#     dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
#     dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
#     dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
#     dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
#     dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
#     dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
#     dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
#     dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
#     dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

#     return dcm

# Helper functions
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    """
    q0, q1, q2, q3 = q
    R = jnp.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R