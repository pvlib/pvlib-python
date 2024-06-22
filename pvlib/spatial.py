"""
Spatial functions for shading and 3D scenes analysis.
"""

from pvlib.tools import sind, cosd
import numpy as np
import shapely as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def _solar_vector(zenith, azimuth):
    """
    Calculate the solar vector in 3D space. Origins from the Sun to the
    observer on Earth characterized by the zenith and azimuth angles.

    Parameters
    ----------
    zenith : numeric
        Solar zenith angle. In degrees [°].
    azimuth : numeric
        Solar azimuth angle. Positive is clockwise from the North in the
        horizontal plane. North=0°, East=90°, South=180°, West=270°.
        In degrees [°].

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
        Unitary solar vector. ``N`=len(zenith)=len(azimuth)``.

    References
    ----------
    .. [1] E. Lorenzo, L. Narvarte, and J. Muñoz, 'Tracking and
       back-tracking', Progress in Photovoltaics: Research and Applications,
       vol. 19, no. 6, pp. 747-753, 2011, :doi:`10.1002/pip.1085`.
    .. [2] Kevin S. Anderson, Adam R. Jensen; Shaded fraction and
       backtracking in single-axis trackers on rolling terrain. J. Renewable
       Sustainable Energy 1 March 2024; 16 (2): 023504.
       :doi:`10.1063/5.0202220`.
    """
    # Eq. (2), [1]; with zenith instead of elevation; coordinate system of [2]
    return np.array(
        [
            sind(zenith) * sind(azimuth),
            sind(zenith) * cosd(azimuth),
            cosd(zenith),
        ]
    )


def _plane_normal_vector(tilt, azimuth):
    """
    Calculate the normal vector of a plane defined by a tilt and azimuth.

    See Eq. (18) of [1]. It has been changed to match system coordinates in
    Fig. 1 of [2].

    Parameters
    ----------
    azimuth : numeric
        Azimuth angle of the plane. Positive is clockwise from the North in
        the horizontal plane. North=0°, East=90°, South=180°, West=270°.
        In degrees [°].
    tilt : numeric
        Tilt angle of the plane. Positive is downwards from the horizontal in
        the direction of ``azimuth``. In degrees [°].

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
        Unitary normal vector of the plane. ``N`=len(azimuth)=len(tilt)``.

    References
    ----------
    .. [1] S. Zainali et al., 'Direct and diffuse shading factors modelling
       for the most representative agrivoltaic system layouts', Applied
       Energy, vol. 339, p. 120981, Jun. 2023,
       :doi:`10.1016/j.apenergy.2023.120981`.
    .. [2] Kevin S. Anderson, Adam R. Jensen; Shaded fraction and
       backtracking in single-axis trackers on rolling terrain. J. Renewable
       Sustainable Energy 1 March 2024; 16 (2): 023504.
       :doi:`10.1063/5.0202220`.
    """
    # Eq. (18) of [1], but coordinate system specified in Fig. 1, [2]
    return np.array(
        [sind(tilt) * sind(azimuth), sind(tilt) * cosd(azimuth), cosd(tilt)]
    )


class FlatSurface:
    """
    Represents a flat surface in 3D space with a given azimuth and tilt and
    boundaries defined by a shapely Polygon.
    Allows to calculate the shading on this surface from other objects,
    both in 2D and 3D.
    In addition, it can

    .. warning::

        This constructor does **not** check the ``azimuth`` and ``tilt`` match
        the ``polygon`` orientation nor the ``polygon`` vertices are coplanar.
        It is the user's responsibility to ensure the surface is correctly
        defined.

    Parameters
    ----------
    tilt : float
        Surface tilt, angle it is inclined with respect to the horizontal
        plane. Tilted downwards ``azimuth``.
        0°=Horizontal, 90°=Vertical. In degrees [°].
    azimuth : float
        Surface azimuth, angle at which it points downwards.
        0°=North, 90°=East, 180°=South, 270°=West. In degrees [°].
    polygon : shapely.Polygon or array[N, 3]
        Shapely Polygon or boundaries to build it.
        Holes are ignored for now.
    roll : float, default 0.0°
        Right-handed rotation around the surface normal vector defined by
        ``tilt`` and ``azimuth``. In degrees [°].
    reference_point : array-like, shape (3,), optional
        Point to use as reference for 2D projections. If not provided, the
        first vertex of the polygon is used.

    References
    ----------
    .. [1] S. Zainali et al., 'Direct and diffuse shading factors modelling
       for the most representative agrivoltaic system layouts', Applied
       Energy, vol. 339, p. 120981, Jun. 2023,
       :doi:`10.1016/j.apenergy.2023.120981`.
    .. [2] Y. Cascone, V. Corrado, and V. Serra, 'Calculation procedure of
       the shading factor under complex boundary conditions', Solar Energy,
       vol. 85, no. 10, pp. 2524-2539, Oct. 2011,
       :doi:`10.1016/j.solener.2011.07.011`.
    .. [3] Kevin S. Anderson, Adam R. Jensen; Shaded fraction and
       backtracking in single-axis trackers on rolling terrain. J. Renewable
       Sustainable Energy 1 March 2024; 16 (2): 023504.
       :doi:`10.1063/5.0202220`.
    """

    @property
    def azimuth(self):
        return self._azimuth

    @property
    def tilt(self):
        return self._tilt

    @property
    def roll(self):
        return self._roll

    @property
    def polygon(self):
        return self._polygon

    @property
    def reference_point(self):
        return self._reference_point

    def __init__(
        self,
        azimuth,
        tilt,
        polygon_boundaries,
        roll=0.0,
        reference_point=None,
    ):
        self._azimuth = np.array(azimuth)
        self._tilt = np.array(tilt)
        self._roll = np.array(roll)
        # works for polygon_boundaries := array[N, 3] | shapely.Polygon
        self._polygon = sp.Polygon(polygon_boundaries)
        # internal 2D coordinates-system to translate projections matrix
        # only defined if needed later on
        self._reference_point = (
            reference_point
            if reference_point is not None
            else self._polygon.exterior.coords[0]
        )
        self._rotation_inverse = None
        self._projected_polygon = None

    def __repr__(self):
        return None

    def _transform_to_2D(self, points):
        """
        Transform a 3D point that **belongs** to the surface,
        to the 2D plane of this surface with respect to the local reference.

        Essentially, it undoes the surface rotations to make the third
        coordinate zero.

        Parameters
        ----------
        point : array-like, shape (N, 3)
            Point to project.

        Returns
        -------
        array-like, shape (N, 2)
            Projected point.

        Raises
        ------
        RuntimeError
            If the 2D projection fails to transform to the surface plane.
            Check the input points belong to the surface.

        References
        ----------
        .. [1] Y. Cascone, V. Corrado, and V. Serra, 'Calculation procedure of
           the shading factor under complex boundary conditions', Solar Energy,
           vol. 85, no. 10, pp. 2524-2539, Oct. 2011,
           :doi:`10.1016/j.solener.2011.07.011`.
        """
        # undo surface rotations to make the third coordinate zero
        _projection = self._get_projection_transform()
        # Section 4.3 in [1], with a plane point
        vertices_2d = _projection.apply(points - self.reference_point)
        if not np.allclose(vertices_2d[:, 2], 0.0, atol=1e-10):
            raise RuntimeError(
                "Non-null third coordinate in 2D projection. "
                + "This should not happen. Please report a bug."
            )
        return np.delete(vertices_2d, 2, axis=1)

    def get_3D_shades_from(self, solar_zenith, solar_azimuth, *others):
        """
        Calculate 3D shades on this surface from other objects.

        3D shade points are guaranteed to be on the projected plane of this
        surface. This method is useful to plot the resulting shade.
        For the shades referred to the surface and a valid area property,
        use the faster method :py:method:`get_2D_shades_from`.

        Shade is clipped to this object boundaries.

        Parameters
        ----------
        solar_zenith : float
            Solar zenith angle. In degrees [°].
        solar_azimuth : float
            Solar azimuth angle. In degrees [°].
        others : FlatSurface or derived
            Obstacles whose shadow will be projected onto this surface.

        Returns
        -------
        shapely.MultiPolygon
            Shapely Polygon objects representing the shades on this surface.
            These shades may overlap.

        Notes
        -----
        This method is based on the algorithm presented in [1]_ to calculate
        the shades, but adds an extra step (a point that belongs to the plane)
        to project the objects onto the surface plane instead of the plane that
        goes through the origin.

        References
        ----------
        .. [1] Y. Cascone, V. Corrado, and V. Serra, 'Calculation procedure of
           the shading factor under complex boundary conditions', Solar Energy,
           vol. 85, no. 10, pp. 2524-2539, Oct. 2011,
           :doi:`10.1016/j.solener.2011.07.011`.
        """
        solar_vec = solar_vec = _solar_vector(  # Eq. (8) -> x,y,z
            solar_zenith, solar_azimuth
        )
        normal_vec = _plane_normal_vector(  # Eq. (18) -> a,b,c
            self._tilt, self._azimuth
        )

        def project_point_to_real_plane(vertex):  # vertex -> Px, Py, Pz
            # Similar to Eq. (20), but takes into account plane position so
            # result belongs to the plane that contains the bounded surface
            t = ((self._reference_point - vertex) @ normal_vec) / (
                solar_vec @ normal_vec
            )
            p_prime = vertex + (t * solar_vec.T).T  # Eq. (19)
            return p_prime

        _polygon = self._polygon  # intersect with 3D surface

        def get_3D_shade_from_flat_surface(other):
            coords_to_project = np.array(other.polygon.exterior.coords[:-1])
            projected_vertices = np.fromiter(
                map(project_point_to_real_plane, coords_to_project),
                dtype=(float, 3),
                count=len(coords_to_project),  # speeds up allocation
            )
            # create shapely shade object and bound it to the surface
            shade = sp.Polygon(projected_vertices).intersection(
                _polygon, grid_size=1e-12
            )
            return shade if isinstance(shade, sp.Polygon) else sp.Polygon()

        return sp.MultiPolygon(map(get_3D_shade_from_flat_surface, others))

    def get_2D_shades_from(
        self, solar_zenith, solar_azimuth, *others, shades_3d=None
    ):
        """
        Calculate 2D shades on this surface from obstacles ``others*``.

        If ``shades`` is provided, the method will combine those shades with
        the ones casted from the objects in ``others`` with the existing ones.
        Use this parameter if you already have the 3D shades to avoid
        recalculating them.

        Parameters
        ----------
        solar_zenith : float
            Solar zenith angle. In degrees [°].
        solar_azimuth : float
            Solar azimuth angle. N=0°, E=90°, S=180°, W=270°. In degrees [°].
        others : FlatSurface or derived
            Obstacles whose shadow will be projected onto this surface.
        shades_3d : shapely.MultiPolygon, optional
            Already calculated 3D shades to combine with the new ones, if any.
            They can be obtained from :py:method:`get_3D_shades_from`.

        Returns
        -------
        shapely.MultiPolygon
            Collection of polygons representing the 2D shades clipped to
            this surface.

        Raises
        ------
        RuntimeError
            If the 2D projection of the shades fails to transform to the
            surface plane. In this case, you must ensure ``shades_3d`` is
            correctly calculated and those shades belong to the object you
            are projecting onto.

        Notes
        -----
        This method is based on the algorithm presented in [1]_ to calculate
        the shades, but adds an extra step to translate the projection objects
        to the plane that goes through the origin, to remove the Z coordinate
        and make the projection in 2D space.

        References
        ----------
        .. [1] Y. Cascone, V. Corrado, and V. Serra, 'Calculation procedure of
           the shading factor under complex boundary conditions', Solar Energy,
           vol. 85, no. 10, pp. 2524-2539, Oct. 2011,
           :doi:`10.1016/j.solener.2011.07.011`.
        """
        shades = self.get_3D_shades_from(solar_zenith, solar_azimuth, *others)
        if shades_3d is not None:
            shades = sp.MultiPolygon((shades.geoms, shades_3d.geoms))

        # and clip the 2D shades to the surface in 2D
        _self_projected_polygon = self.representation_in_2D_space()

        def get_2D_shade_from_flat_surface(other):
            coords_to_project = np.array(other.exterior.coords[:-1])

            # create shapely shade object and bound it to the surface
            projected_vertices_2d = self._transform_to_2D(coords_to_project)
            shade = sp.Polygon(projected_vertices_2d).intersection(
                _self_projected_polygon
            )
            # Return Polygons only, geometries with empty area are omitted
            return shade if isinstance(shade, sp.Polygon) else sp.Polygon()

        return sp.MultiPolygon(
            map(get_2D_shade_from_flat_surface, shades.geoms)
        )

    def _get_projection_transform(self):
        """
        Get the transformation matrix to project points to the 2D plane of
        this surface.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Transformation matrix to project points to the 2D plane of this
            surface.

        References
        ----------
        .. [1] Y. Cascone, V. Corrado, and V. Serra, 'Calculation procedure of
           the shading factor under complex boundary conditions', Solar Energy,
           vol. 85, no. 10, pp. 2524-2539, Oct. 2011,
           :doi:`10.1016/j.solener.2011.07.011`.
        """
        # Section 4.3 in [1], plus another rotation to align the surface:
        # the rotation around the normal vector
        if self._rotation_inverse is None:
            self._rotation_inverse = Rotation.from_euler(
                "zxz",
                [self._azimuth - 180, -self._tilt, -self._roll],
                degrees=True,
            )
        return self._rotation_inverse

    def representation_in_2D_space(self):
        """
        Get the 2D representation of the surface in its local plane, with
        respect to the reference point.

        Returns
        -------
        shapely.Polygon
            2D representation of the surface.

        Notes
        -----
        This method is useful to plot the surface in 2D space and see
        where the shadows are casted.
        """
        if self._projected_polygon is None:
            vertices = np.array(self._polygon.exterior.coords[:-1])
            self._projected_polygon = sp.Polygon(
                self._transform_to_2D(vertices)
            )
        return self._projected_polygon

    def combine_2D_shades(self, *shades):
        """
        Combine overlapping shades into a single one, but keep non-overlapping
        shades separated.

        Parameters
        ----------
        shades : shapely.MultiPolygon
            Shapely MultiPolygon object representing the shades.

        Returns
        -------
        shapely.MultiPolygon
            Combined shades.
        """
        return sp.ops.unary_union(shades)

    def plot(self, ax=None, **kwargs):
        pass  # TODO: implement this method


class RectangularSurface(FlatSurface):
    """
    Represents a rectangular surface in 3D space with a given ``azimuth``,
    ``tilt``, ``axis_tilt`` and a center point. This is a subclass of
    :py:class:`~pvlib.spatial.FlatSurface` handy for rectangular surfaces
    like PV arrays.

    See :py:class:`pvlib.spatial.FlatSurface` for information on methods and
    properties.

    Parameters
    ----------
    center : array-like, shape (3,)
        Center of the surface
    azimuth : float
        Azimuth of the surface. Positive is clockwise from the North in the
        horizontal plane. North=0°, East=90°, South=180°, West=270°.
        In degrees [°].
    tilt : float
        Tilt of the surface, angle it is inclined with respect to the
        horizontal plane. Positive is downwards ``azimuth``.
        In degrees [°].
    axis_tilt : float
        Rotation around the normal vector of the surface. Right-handed.
        In degrees [°].
    width: width of the surface
        For a horizontal surface, the width is parallel to the azimuth
    length: length of the surface
        Perpendicular to the surface azimuth
    reference_point : array-like, shape (3,), optional
        Point to use as reference for 2D projections. If not provided, the
        central point is used.
    """

    def __init__(
        self,
        center,
        azimuth,
        tilt,
        axis_tilt,
        width,
        length,
        reference_point=None,
    ):
        corners = np.array(
            [
                [-length / 2, -width / 2, 0],
                [-length / 2, +width / 2, 0],
                [+length / 2, +width / 2, 0],
                [+length / 2, -width / 2, 0],
            ]
        )
        # rotate corners to match the surface orientation
        # note pvlib convention uses a left-handed azimuth rotation
        # and tilt is defined as the angle from the horizontal plane
        # downwards the azimuth direction (also left-handed)
        # axis_tilt is the rotation around the normal vector, right-handed
        _rotation = Rotation.from_euler(
            "ZXZ", [-azimuth, -tilt, axis_tilt], degrees=True
        )
        polygon = sp.Polygon(_rotation.apply(corners) + center)
        super().__init__(
            azimuth=azimuth,
            tilt=tilt,
            polygon_boundaries=polygon,
            roll=axis_tilt,
            reference_point=reference_point if reference_point else center,
        )

    def plot(self, ax=None, **kwargs):
        """
        Plot the rectangular surface.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes where to plot the surface. If None, a new figure is created.
        **kwargs : dict
            Additional arguments passed to
            :external:ref:`mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        x, y, z = np.hsplit(
            np.array(self._polygon.exterior.coords[:-1]).flatten(order="F"), 3
        )
        return ax.plot_trisurf(
            x, y, z, triangles=((0, 1, 2), (0, 2, 3)), **kwargs
        )
