from __future__ import division

import logging
pvl_logger = logging.getLogger('pvlib')

import numpy as np
import pandas as pd

from pvlib.tools import cosd, sind


def singleaxis(SunZen, SunAz, Latitude=1, 
               AxisTilt=0, AxisAzimuth=0, MaxAngle=90, 
               Backtrack=True, GCR=2.0/7.0):
    """
    Determine the rotation angle of a single axis tracker using the
    equations in [1] when given a particular sun zenith and azimuth angle.
    Backtracking may be specified, and if so, a ground coverage ratio is 
    required.

    Rotation angle is determined in a panel-oriented coordinate system.
    The tracker azimuth AxisAzimuth defines the positive y-axis;
    the positive x-axis is 90 degress clockwise from the y-axis 
    and parallel to the earth surface, and the positive z-axis is 
    normal and oriented towards the sun.
    Rotation angle TrkrTheta indicates tracker position relative to horizontal:
    TrkrTheta = 0 is horizontal, and positive TrkrTheta is a clockwise rotation
    around the y axis in the x, y, z coordinate system.
    For example, if tracker azimuth AxisAzimuth is 180 (oriented south), 
    TrkrTheta = 30 is a rotation of 30 degrees towards the west, 
    and TrkrTheta = -90 is a rotation to the vertical plane facing east.

    Parameters
    ----------
    SunZen : Series
        Apparent (refraction-corrected) zenith angles in decimal degrees. 
    
    SunAz : Series
        Sun azimuth angles in decimal degrees.
    
    Latitude : float
        A value denoting which hempisphere the tracker is
        in. The exact latitude is NOT required, any positive number denotes
        the northern hemisphere, any negative number denotes the southern
        hemisphere, a value of 0 is assumed to be northern hemisphere.
    
    AxisTilt : float
        The tilt of the axis of rotation
        (i.e, the y-axis defined by AxisAzimuth) with respect to horizontal, 
        in decimal degrees.
    
    AxisAzimuth : float
        A value denoting the compass direction along which
        the axis of rotation lies, in decimal degrees. 
    
    MaxAngle : float
        A value denoting the maximum rotation angle, in
        decimal degrees, of the one-axis tracker from its horizontal position
        (horizontal if AxisTilt = 0). 
        A MaxAngle of 90 degrees allows the tracker to rotate to a vertical
        position to point the panel towards a horizon.  
        MaxAngle of 180 degrees allows for full rotation.
    
    Backtrack : bool
        Controls whether the tracker has the
        capability to "backtrack" to avoid row-to-row shading. 
        False denotes no backtrack capability. 
        True denotes backtrack capability. 
    
    GCR : float
        A value denoting the ground coverage ratio of a tracker
        system which utilizes backtracking; i.e. the ratio between the PV
        array surface area to total ground area. A tracker system with modules 2
        meters wide, centered on the tracking axis, with 6 meters between the
        tracking axes has a GCR of 2/6=0.333. If GCR is not provided, a GCR
        of 2/7 is default. GCR must be <=1.

    Returns
    -------
    DataFrame with the following columns:
    
    * TrkrTheta: The rotation angle (Theta) of the tracker.  
        TrkrTheta = 0 is horizontal, and positive rotation angles are
        clockwise.
    * AOI: The angle-of-incidence of direct irradiance onto the
        rotated panel surface.
    * SurfTilt: The angle between the panel surface and the earth
        surface, accounting for panel rotation.
    * SurfAz: The azimuth of the rotated panel, determined by 
        projecting the vector normal to the panel's surface to the earth's
        surface.

    References
    ----------
    [1] Lorenzo, E et al., 2011, "Tracking and back-tracking", Prog. in 
    Photovoltaics: Research and Applications, v. 19, pp. 747-753.
    """
    
    pvl_logger.debug('tracking.singleaxis')
    
    # MATLAB to Python conversion by 
    # Will Holmgren, U. Arizona, March, 2015. @wholmgren
    
    # Calculate sun position x, y, z using coordinate system as in [1], Eq 2.
    # Positive y axis is oriented parallel to earth surface along tracking axis 
    # (for the purpose of illustration, assume y is oriented to the south);
    # positive x axis is orthogonal, 90 deg clockwise from y-axis, and parallel
    # to the earth's surface (if y axis is south, x axis is west); 
    # positive z axis is normal to x,y axes, pointed upward.
    # Equations in [1] assume solar azimuth is relative to reference vector
    # pointed south, with clockwise positive.  Here, the input solar azimuth 
    # is degrees East of North, i.e., relative to a reference vector pointed 
    # north with clockwise positive.
    # Rotate sun azimuth to coordinate system as in [1] 
    # to calculate sun position.
    
    times = SunAz.index
    
    Az = SunAz - 180
    El = 90 - SunZen
    x = cosd(El) * sind(Az)
    y = cosd(El) * cosd(Az)
    z = sind(El)
    
    # translate array azimuth from compass bearing to [1] coord system
    AxisAz = AxisAzimuth - 180

    # translate input array tilt angle axistilt to [1] coordinate system.
    
    # In [1] coordinates, axistilt is a rotation about the x-axis.
    # For a system with array azimuth (y-axis) oriented south, 
    # the x-axis is oriented west, and a positive axistilt is a 
    # counterclockwise rotation, i.e, lifting the north edge of the panel.
    # Thus, in [1] coordinate system, in the northern hemisphere a positive
    # axistilt indicates a rotation toward the equator, 
    # whereas in the southern hemisphere rotation toward the equator is 
    # indicated by axistilt<0.  Here, the input axistilt is
    # always positive and is a rotation toward the equator.

    # Calculate sun position (xp, yp, zp) in panel-oriented coordinate system: 
    # positive y-axis is oriented along tracking axis at panel tilt;
    # positive x-axis is orthogonal, clockwise, parallel to earth surface;
    # positive z-axis is normal to x-y axes, pointed upward.  
    # Calculate sun position (xp,yp,zp) in panel coordinates using [1] Eq 11
    # note that equation for yp (y' in Eq. 11 of Lorenzo et al 2011) is
    # corrected, after conversation with paper's authors.
    
    xp = x*cosd(AxisAz) - y*sind(AxisAz);
    yp = (x*cosd(AxisTilt)*sind(AxisAz) +
          y*cosd(AxisTilt)*cosd(AxisAz) -
          z*sind(AxisTilt))
    zp = (x*sind(AxisTilt)*sind(AxisAz) +
          y*sind(AxisTilt)*cosd(AxisAz) +
          z*cosd(AxisTilt))

    # The ideal tracking angle wid is the rotation to place the sun position 
    # vector (xp, yp, zp) in the (y, z) plane; i.e., normal to the panel and 
    # containing the axis of rotation.  wid = 0 indicates that the panel is 
    # horizontal.  Here, our convention is that a clockwise rotation is 
    # positive, to view rotation angles in the same frame of reference as 
    # azimuth.  For example, for a system with tracking axis oriented south, 
    # a rotation toward the east is negative, and a rotation to the west is 
    # positive.

    # can we use atan2?
    
    # filter to avoid undefined inverse tangent
    
    # angle from x-y plane to projection of sun vector onto x-z plane
    #tmp(xp~=0) = atand(zp./xp)
    # angle from x-y plane to projection of sun vector onto x-z plane
    tmp = np.degrees(np.arctan(zp/xp))  
    #tmp(xp==0 & zp>=0) = 90     # fill in when atan is undefined
    #tmp(xp==0 & zp<0) = -90     # fill in when atan is undefined
    #tmp=tmp(:);                  # ensure tmp is a column vector
    
    # Obtain wid by translating tmp to convention for rotation angles.
    # Have to account for which quadrant of the x-z plane in which the sun 
    # vector lies.  Complete solution here but probably not necessary to 
    # consider QIII and QIV.
    wid = pd.Series(index=times)
    wid[(xp>=0) & (zp>=0)] =  90 - tmp[(xp>=0) & (zp>=0)]  # QI
    wid[(xp<0)  & (zp>=0)] = -90 - tmp[(xp<0)  & (zp>=0)]  # QII
    wid[(xp<0)  & (zp<0)]  = -90 - tmp[(xp<0)  & (zp<0)]   # QIII
    wid[(xp>=0) & (zp<0)]  =  90 - tmp[(xp>=0) & (zp<0)]   # QIV
    #wid=wid(:);                  # ensure wid is a column vector
    
    # filter for sun above panel horizon)
    #u = zp > 0;

    # apply limits to ideal rotation angle
    #wid(~u) = 0;  # set horizontal if zenith<0, sun is below panel horizon
    
    # Account for backtracking; modified from [1] to account for rotation
    # angle convention being used here.
    if Backtrack:
        pvl_logger.debug('applying backtracking')
        Lew = 1/GCR
        temp = np.minimum(Lew*cosd(wid), 1)
        
        # backtrack angle
        # (always positive b/c acosd returns values between 0 and 180)
        wc = np.degrees(np.arccos(temp))
        
        v = wid < 0
        widc = pd.Series(index=times)
        widc[~v] = wid[~v] - wc[~v]; # Eq 4 applied when wid in QI
        widc[v] = wid[v] + wc[v];    # Eq 4 applied when wid in QIV
    else:
        pvl_logger.debug('no backtracking')
        widc = wid
        
    #TrkrTheta[u] = widc[u];
    #TrkrTheta(~u) = 0;    # set to zero when sun is below panel horizon
    #TrkrTheta = TrkrTheta(:);   # ensure column vector format
    TrkrTheta = widc.copy()
    TrkrTheta[TrkrTheta > MaxAngle] = MaxAngle
    TrkrTheta[TrkrTheta < -MaxAngle] = -MaxAngle
    
    # calculate normal vector to panel in panel-oriented x, y, z coordinates
    # y-axis is axis of tracker rotation.  TrkrTheta is a compass angle
    # (clockwise is positive) rather than a trigonometric angle.

    Norm = np.array([sind(TrkrTheta), 
                     np.zeros_like(TrkrTheta),
                     cosd(TrkrTheta)])
    
    # sun position in vector format in panel-oriented x, y, z coordinates
    P = np.array([xp, yp, zp])
    
    # calculate angle-of-incidence on panel
    AOI = np.degrees(np.arccos(np.abs(np.sum(P*Norm, axis=0))))
    #AOI(~u) = 0    # set to zero when sun is below panel horizon
    
    # calculate panel elevation SurfEl and azimuth SurfAz 
    # in a coordinate system where the panel elevation is the 
    # angle from horizontal, and the panel azimuth is
    # the compass angle (clockwise from north) to the projection 
    # of the panel's normal to the earth's surface. 
    # These outputs are provided for convenience and comparison 
    # with other PV software which use these angle conventions.

    # project normal vector to earth surface.
    # First rotate about x-axis by angle -AxisTilt so that y-axis is 
    # also parallel to earth surface, then project.
    
    # Calculate standard rotation matrix
    Rot_x = np.array([[1, 0, 0], 
                      [0, cosd(-AxisTilt), -sind(-AxisTilt)], 
                      [0, sind(-AxisTilt), cosd(-AxisTilt)]])
    
    # temp contains the normal vector expressed in earth-surface coordinates
    # (z normal to surface, y aligned with tracker axis parallel to earth)
    temp = np.dot(Rot_x, Norm) 
    temp = temp.T
    
    # projection to plane tangent to earth surface,
    # in earth surface coordinates
    projNorm = np.array([temp[:,0], temp[:,1], np.zeros_like(temp[:,2])]) 
    tempnorm = np.sqrt(np.nansum(temp**2, axis=1))
    projNormnorm = np.sqrt(np.nansum(projNorm**2, axis=1))

    #SurfAz = 0.*TrkrTheta;
    # calculation of SurfAz
    projNorm = projNorm.T
    SurfAz = np.degrees(np.arctan(projNorm[:,1]/projNorm[:,0]))
    
    # clean up atan when x-coord is zero
    #SurfAz[projNorm(:,1)==0 & projNorm(:,2)>0] =  90;
    #SurfAz[projNorm(:,1)==0 & projNorm(:,2)<0] =  -90;
    # clean up atan when y-coord is zero
    #SurfAz[projNorm(:,2)==0 & projNorm(:,1)>0] =  0;
    #SurfAz[projNorm(:,2)==0 & projNorm(:,1)<0] = 180;
    # correct for QII and QIII
    SurfAz[(projNorm[:,0]<0) & (projNorm[:,1]>0)] += 180 # QII
    SurfAz[(projNorm[:,0]<0) & (projNorm[:,1]<0)] += 180 # QIII

    # at this point SurfAz contains angles between -90 and +270,
    # where 0 is along the positive x-axis,
    # the y-axis is in the direction of the tracker azimuth,
    # and positive angles are rotations from the positive x axis towards
    # the positive y-axis.
    # Adjust to compass angles
    # (clockwise rotation from 0 along the positive y-axis)
    SurfAz[SurfAz<=90] = 90 - SurfAz[SurfAz<=90]
    SurfAz[SurfAz>90] = 450 - SurfAz[SurfAz>90]

    # finally rotate to align y-axis with true north
    if Latitude > 0:
        SurfAz = SurfAz - AxisAzimuth
    else:
        SurfAz = SurfAz - AxisAzimuth - 180
    SurfAz[SurfAz<0] = 360 + SurfAz[SurfAz<0]
    
    #divisor = np.round(tempnorm*projNormnorm*10000)/10000
    #dividend = np.round(temp*projNorm*10000)/10000
    #SurfTilt = 90 - np.degrees(np.arccos(dividend/divisor))
    
    df_out = pd.DataFrame({'AOI':AOI, 'SurfAz':SurfAz, 'SurfTilt':np.nan},
                          index=times)
    
    df_out[SunZen > 90] = np.nan
    
    return df_out