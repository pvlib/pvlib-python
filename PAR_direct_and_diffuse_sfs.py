# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from pvlib.tools import sind, cosd

from itertools import repeat

# %%
x0, y0, z0 = 0, 0, 1 / 2  # m
W, L = 1, 2  # m
corners0 = np.array(
    [
        [x0, y0, z0],
        [x0, y0 + W, z0],
        [x0 + L, y0, z0],
        [x0 + L, y0 + W, z0],
    ]
)


# %%
# def rotation
def Rz(theta):
    return np.array(
        [
            [cosd(theta), -sind(theta), 0],
            [sind(theta), cosd(theta), 0],
            [0, 0, 1],
        ]
    )


def Ry(theta):
    return np.array(
        [
            [cosd(theta), 0, sind(theta)],
            [0, 1, 0],
            [-sind(theta), 0, cosd(theta)],
        ]
    )


def Rx(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, cosd(theta), -sind(theta)],
            [0, sind(theta), cosd(theta)],
        ]
    )


# %%
# rotate corners, general abstraction of row
def rotate_row_with_center_pivot(corners, theta_x, theta_y):
    Rot_x, Rot_y = Rx(theta_x), Ry(theta_y)
    midpoint = np.mean(corners, axis=0)
    Rot = Rot_x @ Rot_y
    return np.array([Rot @ (corner - midpoint) + midpoint for corner in corners])


# %%
# unittest previous function
def test_rotate_row_with_center_pivot():
    corners = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    corners_rotated = rotate_row_with_center_pivot(corners, -90, 0)
    print(corners_rotated)


test_rotate_row_with_center_pivot()


# %%
def plot_corners(corners, fig=None, ax=None, **kwargs) -> plt.Figure:
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection="3d")
    x_ = corners[:, 0].reshape(2, -1)
    y_ = corners[:, 1].reshape(2, -1)
    z_ = corners[:, 2].reshape(2, -1)
    ax.plot_surface(x_, y_, z_, **kwargs)
    return fig


# %%
fig = plot_corners(corners0, alpha=0.5)
fig.show()

# %%
rotated_fixed_tilt = rotate_row_with_center_pivot(corners0, -90, 0)
fig = plot_corners(rotated_fixed_tilt, alpha=0.5)
fig.show()


# %%
def solar_vector(zenith, azimuth):
    # Eq. (8), but in terms of zenith instead of elevation
    return np.array(
        [
            sind(zenith) * sind(azimuth),
            sind(zenith) * cosd(azimuth),
            cosd(zenith),
        ]
    )


def normal_vector_of_row(tilt, azimuth):
    # Eq. (18)
    return np.array(
        [
            sind(tilt) * cosd(azimuth),
            sind(tilt) * sind(azimuth),
            cosd(tilt),
        ]
    )


def corners_projection_onto_plane(
    corners, solar_zenith, solar_azimuth, plane_tilt, plane_azimuth
):
    x, y, z = solar_vector(solar_zenith, solar_azimuth)
    a, b, c = normal_vector_of_row(plane_tilt, plane_azimuth)
    def projection(corner):
        Px, Py, Pz = corner
        # Eq. (20)
        t = - (a*Px+b*Py+c*Pz) / (a*x+b*y+c*z)
        # Eq. (19)
        p_prime = np.array([Px + x*t, Py + y*t, Pz + z*t])
        return p_prime
    return np.array([projection(corner) for corner in corners])


# %%
# set hypothetical values
solar_zenith = 30
solar_azimuth = 180
row_tilt = 30
row_azimuth = 180
plane_tilt = -10
plane_azimuth = 90

# create the row
corners1 = rotate_row_with_center_pivot(corners0, row_tilt, 0)
corners_projected = corners_projection_onto_plane(corners1, solar_zenith, solar_azimuth, plane_tilt, plane_azimuth)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 2)
ax.set_ylim(0, 3)
ax.set_zlim(0, 1)
_ = plot_corners(corners1, ax=ax, alpha=0.5, color="grey")
_ = plot_corners(corners_projected, ax=ax, alpha=0.5, color="brown")
fig.show()

# %%
# Compare corners_projected with corners after rotation to new coordinate systems of the projection plane, to remove the third coordinate

def from_3d_plane_to_2d(points, tilt, azimuth):
    # Section 4.3 in [2]
    R_z = Rz(90 + azimuth)
    R_x = Rx(tilt)
    rot = (R_z @ R_x).T
    if points.ndim == 1:
        points = points.reshape(1, -1)
    new_points = np.fromiter(map(rot.__matmul__, points), dtype=(float, 3))
    assert np.allclose(new_points[:, 2], 0), "The third coordinate should be zero, check input parameters are consistent with the projection plane."
    return np.delete(new_points, 2, axis=1)


new_points = from_3d_plane_to_2d(corners_projected, plane_tilt, plane_azimuth)

print(new_points)

# %%
