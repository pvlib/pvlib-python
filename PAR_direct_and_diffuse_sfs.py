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
    return np.array(
        [Rot @ (corner - midpoint) + midpoint for corner in corners]
    )


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
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners)
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
        t = -(a * Px + b * Py + c * Pz) / (a * x + b * y + c * z)
        # Eq. (19)
        p_prime = np.array([Px + x * t, Py + y * t, Pz + z * t])
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
corners_projected = corners_projection_onto_plane(
    corners1, solar_zenith, solar_azimuth, plane_tilt, plane_azimuth
)

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
    assert np.allclose(
        new_points[:, 2], 0
    ), "The third coordinate should be zero, check input parameters are consistent with the projection plane."
    return np.delete(new_points, 2, axis=1)


new_points = from_3d_plane_to_2d(corners_projected, plane_tilt, plane_azimuth)

print(new_points)

# %%
#
##########################
# OOP Implementation
##########################
import numpy as np
import pandas as pd
import shapely as sp
from pvlib.tools import sind, cosd, acosd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


def atan2d(y, x):
    return np.degrees(np.arctan2(y, x))


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


def solar_vector(zenith, azimuth):
    # Eq. (8), but in terms of zenith instead of elevation
    return np.array(
        [
            sind(zenith) * sind(azimuth),
            sind(zenith) * cosd(azimuth),
            cosd(zenith),
        ]
    ).T


def normal_vector_of_surface(tilt, azimuth):
    # Eq. (18)
    return np.array(
        [
            sind(tilt) * cosd(azimuth),
            sind(tilt) * sind(azimuth),
            cosd(tilt),
        ]
    ).T


# %%
class FlatSurface:
    def __init__(
        self, azimuth, tilt, polygon_boundaries
    ):
        """

        .. warning::

            This constructor does not make any check on the input parameters.
            Use any of the class methods ``from_*`` to create an object.

        Parameters
        ----------
        azimuth : float
            Surface azimuth, angle at which it points downwards. 0° is North, 90° is East, 180° is South, 270° is West. In degrees [°].
        tilt : float
            Surface tilt, angle at which it is inclined with respect to the horizontal plane. Tilted downwards ``azimuth``. 0° is horizontal, 90° is vertical. In degrees [°].
        polygon : shapely.LinearRing or array[N, 3]
            Shapely Polygon or boundaries to build it. If you need to specify holes, provide your custom LinearRing.
        """
        self.azimuth = azimuth
        self.tilt = tilt
        # works for polygon_boundaries := array[N, 3] | shapely.Polygon
        self.polygon = sp.LinearRing(polygon_boundaries)
        # TODO: REMOVABLE?
        #  representative point to calculate if obstacles are in front or behind
        # if point_on_surface is None:
        #     self.belonging_point = self.polygon.point_on_surface()
        # else:
        #     self.belonging_point = point_on_surface
        # self.belonging_point = np.array(self.belonging_point)
        # internal 2D coordinates-system to translate projections matrix
        # only defined if needed later on
        self._projection_matrix = None
        self._projected_polygon = None

    def get_3D_shades_from(self, solar_zenith, solar_azimuth, *others):
        # project ``others`` onto this shaded surface
        # return the shade shapely object
        solar_vec = solar_vector(solar_zenith, solar_azimuth)  # x,y,z
        normal_vec = normal_vector_of_surface(self.tilt, self.azimuth)  # a,b,c

        def point_projection(vertex):  # vertex := Px, Py, Pz
            t = -(normal_vec @ vertex) / (solar_vec @ normal_vec)  # Eq. (20)
            p_prime = vertex + (t * solar_vec.T).T  # Eq. (19)
            return p_prime

        # undo surface rotations to make the third coordinate zero
        _projection_matrix = self._get_projection_matrix_matrix()
        _projected_polygon = self._get_self_projected_polygon()

        projected_vertices = np.fromiter(
            map(point_projection, other.polygon.coords[:-1]), dtype=(float, 3)
        )

        def get_3D_shade_from_flat_surface(other):
            # Section 4.3 in [2]
            if projected_vertices.ndim == 1:
                projected_vertices = projected_vertices.reshape(1, -1)
            vertices_2d = np.fromiter(
                map(_projection_matrix.__matmul__, projected_vertices),
                dtype=(float, 3),
            )
            if not np.allclose(vertices_2d[:, 2], 0.0, atol=1e-10):
                raise RuntimeError(
                    "The third coordinate should be zero, check input "
                    + "parameters are consistent with the projection plane."
                    + " If you see this, the error is on me. I fkd up."
                )
            vertices_2d = np.delete(vertices_2d, 2, axis=1)
            # create a 2D polygon
            polygon = sp.Polygon(vertices_2d).intersection(_projected_polygon)
            return polygon

        return tuple(map(get_shade_from_flat_surface, others))

    def _get_projection_matrix_matrix(self):
        if self._projection_matrix is None:
            self._projection_matrix = (Rz(90 + self.azimuth) @ Rx(self.tilt)).T
        return self._projection_matrix

    def _get_self_projected_polygon(self):
        if self._projected_polygon is None:
            _projection_matrix = self._get_projection_matrix_matrix()
            self._projected_polygon = sp.Polygon(
                (
                    _projection_matrix @ vertex
                    for vertex in self.polygon.coords[:-1]
                )
            )
        return self._projected_polygon


# %%
class RectangularSurface(FlatSurface):
    def __init__(
        self, center, surface_azimuth, surface_tilt, axis_tilt, width, length
    ):
        """

        .. warning::

            This constructor does not make any check on the input parameters.
            Use any of the class methods ``from_*`` to create an object.

        center: center of the surface
        surface_azimuth: azimuth of the surface
        surface_tilt: tilt of the surface
        width: width of the surface
        length: length of the surface
            TODO: which one is perpendicular to azimuth?


        """
        self.center = np.array(center)
        x_c, y_c, z_c = center
        self.surface_azimuth = surface_azimuth
        self.surface_tilt = surface_tilt
        self.axis_tilt = axis_tilt
        self.width = width
        self.length = length
        corners = np.array(
            [
                [x_c - length / 2, y_c - width / 2, z_c],
                [x_c - length / 2, y_c + width / 2, z_c],
                [x_c + length / 2, y_c + width / 2, z_c],
                [x_c + length / 2, y_c - width / 2, z_c],
            ]
        )
        # rotate corners to match the surface orientation
        # note pvlib convention uses a left-handed azimuth rotation
        rot = Rz(180 - surface_azimuth) @ Ry(axis_tilt) @ Rx(surface_tilt)
        self.shapely_obj = sp.LinearRing(
            [rot @ (corner - self.center) + self.center for corner in corners]
        )
        tilt, azimuth = self._calc_surface_tilt_and_azimuth(rot)
        super().__init__(azimuth, tilt, self.shapely_obj)
        self._projection_matrix = rot.T

    @classmethod
    def _calc_surface_tilt_and_azimuth(cls, rotation_matrix):
        # tz as in K. Anderson and M. Mikofski paper, somefig
        tz_x, tz_y, tz_z = rotation_matrix[:, 2]  # := rot @ [0, 0, 1].T
        tilt = acosd(tz_z)
        azimuth = atan2d(tz_y, tz_x)
        return tilt, azimuth

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        x, y, z = np.hsplit(
            np.array(self.shapely_obj.coords[:-1]).flatten(order="F"), 3
        )
        ax.plot_trisurf(x, y, z, triangles=((0, 1, 2), (0, 2, 3)), **kwargs)
        return ax


# %%
# Paper Examples
# ==============
#
# +------------------------+----------+-------------+----------+-------+
# | Input parameters       | Vertical | Single-axis | Two-axis | Units |
# +========================+==========+=============+==========+=======+
# | Panel width            |        1 |           1 |        1 |   [m] |
# +------------------------+----------+-------------+----------+-------+
# | Panel length           |        2 |           2 |        2 |   [m] |
# +------------------------+----------+-------------+----------+-------+
# | Number of panels       |       40 |          40 |       40 |   [-] |
# +------------------------+----------+-------------+----------+-------+
# | Total panel area       |       80 |          80 |       80 |  [m²] |
# +------------------------+----------+-------------+----------+-------+
# | Number of rows         |        2 |           2 |        2 |   [-] |
# +------------------------+----------+-------------+----------+-------+
# | Row spacing            |       10 |          10 |       10 |   [m] |
# +------------------------+----------+-------------+----------+-------+
# | Row length             |       20 |          20 |       20 |   [m] |
# +------------------------+----------+-------------+----------+-------+
# | Crop area              |      200 |         200 |      200 |  [m²] |
# +------------------------+----------+-------------+----------+-------+
# | Pitch                  |        - |           - |        2 |   [m] |
# +------------------------+----------+-------------+----------+-------+
# | Height                 |        0 |           3 |        3 |   [m] |
# +------------------------+----------+-------------+----------+-------+
# | Fixed tilt angle       |       90 |           - |        - |   [°] |
# +------------------------+----------+-------------+----------+-------+
# | Azimuth angle          |        0 |           0 |        0 |   [°] |
# +------------------------+----------+-------------+----------+-------+
# | Maximum tilt angle     |        - |          60 |       60 |   [°] |
# +------------------------+----------+-------------+----------+-------+
# | Minimum tilt angle     |        - |         -60 |      -60 |   [°] |
# +------------------------+----------+-------------+----------+-------+
#
#

# Kärrbo Prästgård, Västerås, Sweden
latitude, longitude, altitude = 59.6099, 16.5448, 20  # °N, °E, m

spring_equinox = pd.date_range("2021-03-20", periods=24, freq="H")
summer_solstice = pd.date_range("2021-06-21", periods=24, freq="H")
fall_equinox = pd.date_range("2021-09-22", periods=24, freq="H")
winter_solstice = pd.date_range("2021-12-21", periods=24, freq="H")
dates = (
    spring_equinox.union(summer_solstice)
    .union(fall_equinox)
    .union(winter_solstice)
)

solar_azimuth = 120
solar_zenith = [60, 80]

# %%
# Fixed Tilt
# ----------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
field = RectangularSurface([20 / 2, 10 / 2, 0], 0, 0, 0, 10, 20)
field.plot(ax=ax, color="forestgreen", alpha=0.5)
pv_row1 = RectangularSurface([20 / 2, 0, 2 / 2], 180, 90, 0, 2, 20)
pv_row1.plot(ax=ax, color="darkblue", alpha=0.5)
pv_row2 = RectangularSurface([20 / 2, 10, 2 / 2], 180, 90, 0, 2, 20)
pv_row2.plot(ax=ax, color="darkblue", alpha=0.5)



# %%
shades = field.get_shades_by(solar_zenith, solar_azimuth, pv_row1, pv_row2)
# %%
