"""
Lab-frame coordinate transformation for OPALX HDF5 phase-space files.

OPALX stores particle positions/momenta in the beam reference frame
(centred on the reference particle, oriented along the beam direction).
The step attributes RefPartR and RefPartP give the reference particle's
position and momentum in the global lab frame.

TaitBryantAngles is stored in the H5 file but is currently NOT written
by OPALX (H5PartWrapperForPT.cpp:329 -- the initialisation is commented
out). When non-zero angles are present the quaternion-based rotation is
used; otherwise the rotation is derived from RefPartP.

Transform:
    R_lab[i] = M @ R_beam[i] + RefPartR
    P_lab[i] = M @ P_beam[i]              (beam-relative momenta)
    P_lab[i] = M @ P_beam[i] + RefPartP   (absolute momenta)

where M is the 3x3 rotation matrix taking beam-frame vectors to lab frame.
"""

import numpy as np


def _quat_multiply(q1, q2):
    """Quaternion multiplication. Both in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _rotation_matrix_from_quat(q):
    """3x3 rotation matrix from unit quaternion (w, x, y, z).

    Matches the formula used in OPALX Quaternion.hpp.
    M takes a beam-frame vector to lab frame (rotateTo convention).
    """
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),  2*(-w*z + x*y),   2*( w*y + x*z)],
        [2*( w*z + x*y),     1 - 2*(x*x + z*z), 2*(-w*x + y*z)],
        [2*(-w*y + x*z),     2*( w*x + y*z),    1 - 2*(x*x + y*y)],
    ])


def build_rotation_from_tait_bryant(angles):
    """Rotation matrix from OPALX Tait-Bryant angles [theta, phi, psi].

    Reconstruction follows H5PartWrapperForPT.cpp:
        rotTheta = (cos(t/2), 0,       sin(t/2), 0)       # Y-axis
        rotPhi   = (cos(p/2), sin(p/2), 0,       0)       # X-axis
        rotPsi   = (cos(s/2), 0,        0,       sin(s/2))# Z-axis
        q        = rotTheta * (rotPhi * rotPsi)

    Returns identity matrix if all angles are zero (guards against the
    current OPALX bug where TaitBryantAngles is written uninitialised).
    """
    theta, phi, psi = angles
    if theta == 0.0 and phi == 0.0 and psi == 0.0:
        return np.eye(3)

    q_theta = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
    q_phi   = np.array([np.cos(phi   / 2), np.sin(phi   / 2), 0.0, 0.0])
    q_psi   = np.array([np.cos(psi   / 2), 0.0, 0.0, np.sin(psi   / 2)])

    q = _quat_multiply(q_theta, _quat_multiply(q_phi, q_psi))
    return _rotation_matrix_from_quat(q)


def build_rotation_from_ref_p(ref_p):
    """Rotation matrix that maps beam-frame z-axis onto the RefPartP direction.

    The beam local z-axis corresponds to the direction of beam propagation.
    In the lab frame this direction is given by RefPartP. Roll around the
    beam axis is set to zero (no information in the H5 file for this).

    Handles degenerate cases:
      RefPartP ∥ (0,0,1)  → identity
      RefPartP ∥ (0,0,-1) → 180-degree rotation around X
    """
    norm = np.linalg.norm(ref_p)
    if norm == 0.0:
        return np.eye(3)

    d = ref_p / norm
    z = np.array([0.0, 0.0, 1.0])

    cos_angle = np.clip(np.dot(z, d), -1.0, 1.0)

    if cos_angle > 1.0 - 1e-12:
        return np.eye(3)

    if cos_angle < -1.0 + 1e-12:
        # 180-degree rotation around X axis
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(cos_angle)

    # Rodrigues' rotation formula
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * K @ K


def get_step_attrs(step_group):
    """Read RefPartR, RefPartP, TaitBryantAngles from a step group's attributes."""
    def read_vec(name):
        if name in step_group.attrs:
            return np.asarray(step_group.attrs[name], dtype=float).reshape(3)
        return None

    ref_r = read_vec("RefPartR")
    ref_p = read_vec("RefPartP")
    tait  = read_vec("TaitBryantAngles")
    return ref_r, ref_p, tait


def get_rotation_matrix(ref_p, tait_bryant):
    """Choose and build the best available rotation matrix.

    Prefers TaitBryantAngles when non-zero (future-proof once OPALX
    writes them correctly). Falls back to RefPartP-derived rotation.
    """
    if tait_bryant is not None and not np.allclose(tait_bryant, 0.0):
        return build_rotation_from_tait_bryant(tait_bryant)

    if ref_p is not None:
        return build_rotation_from_ref_p(ref_p)

    return np.eye(3)


def transform_to_lab(positions, momenta, step_group, absolute_momentum=False):
    """Transform particle positions and momenta to the lab frame.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions in beam frame (x, y, z columns).
    momenta : ndarray, shape (N, 3) or None
        Particle momenta in beam frame (px, py, pz columns), or None.
    step_group : h5py.Group
        The HDF5 step group (attributes are read from here).
    absolute_momentum : bool
        If True, add RefPartP to transformed momenta to get absolute lab
        momenta. If False, momenta are beam-relative but rotated to lab
        orientation.

    Returns
    -------
    positions_lab : ndarray, shape (N, 3)
    momenta_lab   : ndarray, shape (N, 3) or None
    """
    ref_r, ref_p, tait = get_step_attrs(step_group)
    M = get_rotation_matrix(ref_p, tait)

    # positions: rotate then translate
    positions_lab = (M @ positions.T).T
    if ref_r is not None:
        positions_lab += ref_r

    # momenta: rotate only (optionally add RefPartP)
    momenta_lab = None
    if momenta is not None:
        momenta_lab = (M @ momenta.T).T
        if absolute_momentum and ref_p is not None:
            momenta_lab += ref_p

    return positions_lab, momenta_lab
