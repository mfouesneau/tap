""" Rudimentary coordinate transformations and operations """

import numpy as np
from math import pi, atan2, cos, sin, radians, sqrt, degrees

# Obliquity of the Ecliptic (arcsec)
obliquityOfEcliptic = radians(84381.41100 / 3600.0)

# Galactic pole in ICRS coordinates (see Hipparcos Explanatory Vol 1 section
# 1.5, and Murray, 1983, section 10.2)
# J2000
alpha_galactic_pole = radians(192.85948)
delta_galactic_pole = radians(27.12825)
# The galactic longitude of the ascending node of the galactic plane on the
# equator of ICRS (see Hipparcos Explanatory Vol 1 section 1.5, and Murray,
# 1983, section 10.2)
omega = radians(32.93192)


def elementaryRotationMatrix(axis, angle):
    """
    Construct the rotation matrix associated with the roation of given angle
    along given x,y or z axis-vector.

    Parameters
    ----------
    axis: str
        Axis around which to rotate ("x", "y", or "z")
    angle: float
        the rotation angle in radians

    Returns
    -------
    mat: ndarray
        The rotation matrix
    """
    if (axis.lower() == "x"):
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, cos(angle), sin(angle)],
                         [0.0, -sin(angle), cos(angle)]])
    elif (axis.lower() == "y"):
        return np.array([[cos(angle), 0.0, -sin(angle)],
                         [0.0, 1.0, 0.0],
                         [sin(angle), 0.0, cos(angle)]])
    elif (axis.lower() == "z"):
        return np.array([[cos(angle), sin(angle), 0.0],
                         [-sin(angle), cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    else:
        raise Exception("Unknown rotation axis " + axis + "!")


# Rotation matrix for the transformation from ICRS to Galactic coordinates. See
# equation (4.25) in chapter 4.5 of "Astrometry for Astrophysics", 2012, van
# Altena et al.
_matA = elementaryRotationMatrix("z", pi / 2.0 + alpha_galactic_pole)
_matB = elementaryRotationMatrix("x", pi / 2.0 - delta_galactic_pole)
_matC = elementaryRotationMatrix("z", -omega)
_rotationMatrixIcrsToGalactic = np.dot(_matC, np.dot(_matB, _matA))

# Rotation matrix for the transformation from Galactic to ICRS coordinates.
_rotationMatrixGalacticToIcrs = np.transpose(_rotationMatrixIcrsToGalactic)

# Rotation matrix for the transformation from Ecliptic to ICRS coordinates.
_rotationMatrixEclipticToIcrs = elementaryRotationMatrix("x", -1 * (obliquityOfEcliptic))

# Rotation matrix for the transformation from ICRS to Ecliptic coordinates.
_rotationMatrixIcrsToEcliptic = np.transpose(_rotationMatrixEclipticToIcrs)

# Rotation matrix for the transformation from Galactic to Ecliptic coordinates.
_rotationMatrixGalacticToEcliptic = np.dot(_rotationMatrixIcrsToEcliptic,_rotationMatrixGalacticToIcrs)

# Rotation matrix for the transformation from Ecliptic to Galactic coordinates.
_rotationMatrixEclipticToGalactic = np.transpose(_rotationMatrixGalacticToEcliptic)

# Mappings between transformations and matrices
_rotationMatrix = dict(
    GAL2ICRS=_rotationMatrixGalacticToIcrs,
    ICRS2GAL=_rotationMatrixIcrsToGalactic,
    ECL2ICRS=_rotationMatrixEclipticToIcrs,
    ICRS2ECL=_rotationMatrixIcrsToEcliptic,
    GAL2ECL=_rotationMatrixGalacticToEcliptic,
    ECL2GAL=_rotationMatrixEclipticToGalactic
)

# Mapping in/out transformations
_transformationString = dict(
    GAL2ICRS=("galactic", "ICRS"),
    ICRS2GAL=("ICRS","galactic"),
    ECL2ICRS=("ecliptic","ICRS"),
    ICRS2ECL=("ICRS","ecliptic"),
    GAL2ECL=("galactic","ecliptic"),
    ECL2GAL=("ecliptic","galactic")
)


def _np_sphericalToCartesian(r, phi, theta):
    ctheta = np.cos(theta)
    x = r * np.cos(phi) * ctheta
    y = r * np.sin(phi) * ctheta
    z = r * np.sin(theta)
    return x, y, z


def _math_sphericalToCartesian(r, phi, theta):
    ctheta = cos(theta)
    x = r * cos(phi) * ctheta
    y = r * sin(phi) * ctheta
    z = r * sin(theta)
    return x, y, z


def _np_cartesianToSpherical(x, y, z):
    rCylSq = x * x + y * y
    r = np.sqrt(rCylSq + z * z)
    if np.any(r == 0.0):
        raise Exception("Error: one or more of the points is at distance zero.")
    return r, np.arctan2(y, x), np.arctan2(z, np.sqrt(rCylSq))


def _math_cartesianToSpherical(x, y, z):
    rCylSq = x * x + y * y
    r = sqrt(rCylSq + z * z)
    if r == 0.0:
        raise Exception("Error: point is at distance zero.")
    return r, atan2(y, x), atan2(z, sqrt(rCylSq))


def sphericalToCartesian(r, phi, theta):
    """
    Convert spherical to Cartesian coordinates.

    The input can be scalars or 1-dimensional numpy arrays.  Note that the angle
    coordinates follow the astronomical convention of using elevation
    (declination, latitude) rather than its complement (pi/2-elevation), where
    the latter is commonly used in the mathematical treatment of spherical
    coordinates.

    Parameters
    ----------
    r     - length of input Cartesian vector.
    phi   - longitude-like angle (e.g., right ascension, ecliptic longitude) in radians
    theta - latitide-like angle (e.g., declination, ecliptic latitude) in radians

    Returns
    -------
    x, y, z: tuple
    The Cartesian vector components x, y, z
    """
    if hasattr(r, '__iter__'):
        return _np_sphericalToCartesian(r, phi, theta)
    else:
        return _math_sphericalToCartesian(r, phi, theta)


def cartesianToSpherical(x, y, z):
    """
    Convert Cartesian to spherical coordinates. The input can be scalars or
    1-dimensional numpy arrays.  Note that the angle coordinates follow the
    astronomical convention of using elevation (declination, latitude) rather
    than its complement (pi/2-elevation), which is commonly used in the
    mathematical treatment of spherical coordinates.

    Parameters
    ----------
    x - Cartesian vector component along the X-axis
    y - Cartesian vector component along the Y-axis
    z - Cartesian vector component along the Z-axis

    Returns
    -------
    The spherical coordinates r=sqrt(x*x+y*y+z*z), longitude phi, latitude theta.

    NOTE THAT THE LONGITUDE ANGLE IS BETWEEN -PI AND +PI. FOR r=0 AN EXCEPTION IS RAISED.
    """
    if hasattr(x, '__iter__'):
        return _np_cartesianToSpherical(x, y, z)
    else:
        return _math_cartesianToSpherical(x, y, z)


def apply_transformation(name, phi, theta, use_degrees=True):
    """ apply transformation coordinates

    Parameters
    ----------
    name: str
        transformation name
    phi: float or sequence
        first coordinate in radians
    theta: float or sequence
        second coordinate in radian
    use_degrees: bool
        set to convert to degree units for input and outputs

    Returns
    -------
    a: float or ndarray
        transformed first coordinate in radians
    b: float or array
        transformed second coordinate in radians
    """
    if hasattr(phi, '__iter__'):
        r = np.ones_like(phi)
        if use_degrees:
            x, y, z = _np_sphericalToCartesian(r,
                                               np.deg2rad(phi),
                                               np.deg2rad(theta))
        else:
            x, y, z = _np_sphericalToCartesian(r, phi, theta)
        mat = _rotationMatrix[name]
        xrot, yrot, zrot = np.dot(mat, [x,y,z])
        _, a, b = _np_cartesianToSpherical(xrot, yrot, zrot)
        if use_degrees:
            a = np.rad2deg(a)
            b = np.rad2deg(b)
    else:
        r = 1.
        if use_degrees:
            x, y, z = _math_sphericalToCartesian(r, radians(phi),
                                                 radians(theta))
        else:
            x, y, z = _math_sphericalToCartesian(r, phi, theta)
        mat = _rotationMatrix[name]
        xrot, yrot, zrot = np.dot(mat, [x,y,z])
        _, a, b = _math_cartesianToSpherical(xrot, yrot, zrot)
        if use_degrees:
            a = degrees(a)
            b = degrees(b)
    return a, b

def spherical_distance(ra1, dec1, ra2, dec2, use_degrees=True):
    """ Compute the angular distance between 2 points on the sphere in degrees

    Parameters
    ----------
    ra1: float or array
        first longitude  in degrees
    dec1: float or array
        first latitude in degrees
    ra2: float or array
        second longitude  in degrees
    dec2: float or array
        second latitude in degrees
    use_degrees: bool
        set to convert to degree units for input and outputs

    Returns
    -------
    dist: float
        distance in degrees
    """
    if use_degrees:
        return np.rad2deg(spherical_distance(
            np.deg2rad(ra1), np.deg2rad(dec1),
            np.deg2rad(ra2), np.deg2rad(dec2)),
            use_degrees=False
        )
    else:
        return 2. * np.arcsin(
            np.sqrt(
                np.sin((dec1 - dec2) / 2)  * np.sin( (dec1 - dec2) / 2)
                + np.cos(dec1) * np.cos(dec2) * (
                    np.sin( (ra1 - ra2) / 2) * np.sin( (ra1 - ra2) / 2) )
            ))


def get_FK5PrecessMatrix(begEpoch, endEpoch):
    """ Return the matrix of precession between two epochs (IAU 1976, FK5)

    Though the matrix method itself is rigorous, the precession
    angles are expressed through canonical polynomials which are
    valid only for a limited time span.  There are also known
    errors in the IAU precession rate.
    The absolute accuracy of the present formulation is
    better than 0.1 arcsec from 1960AD to 2040AD,
    better than 1 arcsec from 1640AD to 2360AD,
    and remains below 3 arcsec for the whole of the period 500BC to 3000AD.

    The errors exceed 10 arcsec outside the range 1200BC to 3900AD, exceed
    100 arcsec outside 4200BC to 5600AD and exceed 1000 arcsec outside 6800BC
    to 8200AD.

    References:
    Lieske,J.H., 1979. Astron.Astrophys.,73,282. equations (6) & (7), p283.
    Kaplan,G.H., 1981. USNO circular no. 163, pA2.

    Parameters
    ----------
    begEpoch: float
        beginning Julian epoch (e.g. 2000 for J2000)
    endEpoch: float
        ending Julian epoch

    Returns
    -------
    pMat: ndarray (3,3)
        the precession matrix as a 3x3 matrix, where pos(endEpoch) = rotMat * pos(begEpoch)

     """
    # Interval between basic epoch J2000.0 and beginning epoch (JC)
    t0 = (begEpoch - 2000.0) / 100.0      # origin J2000

    # Interval over which precession required (JC)
    dt = (endEpoch - begEpoch) / 100.0   # centuries

    # Euler angles
    RadPerDeg  = pi / 180.0              # radians per degree
    ArcSecPerDeg = 60.0 * 60.0                # arcseconds per degree
    RadPerArcSec = RadPerDeg / ArcSecPerDeg   # radians per arcsec
    tas2r = dt * RadPerArcSec
    w = 2306.2181 + (1.39656 - 0.000139 * t0) * t0   # arcsec

    zeta = (w + ((0.30188 - 0.000344 * t0) + 0.017998 * dt) * dt) * tas2r
    z = (w + ((1.09468 + 0.000066 * t0) + 0.018203 * dt) * dt) * tas2r
    theta = ((2004.3109 + ( - 0.85330 - 0.000217 * t0) * t0) + (( - 0.42665 - 0.000217 * t0) - 0.041833 * dt) * dt) * tas2r

    # Rotation matrix
    mat = np.dot(np.dot(elementaryRotationMatrix("z", z),
                        elementaryRotationMatrix("y", theta)),
                 elementaryRotationMatrix("z", zeta))
    return mat


def fk5xyz_applyPrecession(pos, pm, fromDate, toDate):
    """ Apply precession and proper motion of stars

    References:
    "The Astronomical Almanac" for 1987, page B39
    P.T. Wallace's prec routine

    Parameters
    ----------
    pos: ndarray
        initial mean FK5 cartesian position (au)
    pm: ndarray
        initial mean FK5 cartesian velocity (au per Jul. year)
    fromDate: float
        date of initial coordinates (Julian epoch)
    toDate: float   date to which to precess (Julian epoch)

    Returns
    -------
    newpos: ndarray
        final mean FK5 cartesian position (au)
    newpm: ndarray
        final mean FK5 cartesian velocity (au per Julian year)
    """
    # compute new precession constants
    rotMat = get_FK5PrecessMatrix(fromDate, toDate)
    # correct for velocity (proper motion and radial velocity)
    dt = toDate - fromDate
    tempP = pos + pm * dt
    # precess position and velocity
    newpos = np.dot(rotMat, tempP)
    newpm = np.dot(rotMat, pm)
    return newpos, newpm


def apply_precession_with_pm(phi, theta, muphi, mutheta, fromDate, toDate,
                             use_degrees=True):
    """ Apply precession and proper motion of stars

    References:
    "The Astronomical Almanac" for 1987, page B39
    P.T. Wallace's prec routine

    Parameters
    ----------
    phi: ndarray
        first coordinate in radians (or degrees if use_degrees)
    theta: ndarray
        second coordinate in radians (or degrees if use_degrees)
    muphi: ndarray
        proper motion along phi in radians (or degrees if use_degrees)
    mutheta: ndarray
        proper motion along theta in radians (or degrees if use_degrees)
    use_degrees: bool
        if set, takes input angles in degrees and produces outputs also in degrees

    Returns
    -------
    newpos: ndarray
        final positions
    newpm: ndarray
        final velocities
    """
    r = np.ones_like(phi)

    # compute new precession constants
    rotMat = get_FK5PrecessMatrix(fromDate, toDate)
    dt = toDate - fromDate

    if use_degrees:
        xyz = sphericalToCartesian(r, np.deg2rad(phi), np.deg2rad(theta))
        vxyz = sphericalToCartesian(r, np.deg2rad(muphi), np.deg2rad(mutheta))
    else:
        xyz = sphericalToCartesian(r, phi, theta)
        vxyz = sphericalToCartesian(r, muphi, mutheta)

    tempV = np.array(vxyz).T
    tempP = np.array(xyz).T + tempV * dt

    xyzrot = np.array([np.dot(rotMat, k) for k in tempP])
    vxyzrot = np.array([np.dot(rotMat, k) for k in tempV])
    #xyzrot = np.dot(rotMat, tempP)
    #vxyzrot = np.dot(rotMat, tempV)

    rab = cartesianToSpherical(xyzrot[:, 0], xyzrot[:, 1], xyzrot[:, 2])
    vab = cartesianToSpherical(vxyzrot[:, 0], vxyzrot[:, 1], vxyzrot[:, 2])

    newpos = np.array([rab[1], rab[2]]).T
    newvel = np.array([vab[1], vab[2]]).T

    if use_degrees:
        return np.rad2deg(newpos), np.rad2deg(newvel)
    else:
        return newpos, newvel


def apply_precession(phi, theta, fromDate, toDate, use_degrees=True):
    """ Apply precession and proper motion of stars

    References:
    "The Astronomical Almanac" for 1987, page B39
    P.T. Wallace's prec routine

    Parameters
    ----------
    phi: ndarray
        first coordinate in radians (or degrees if use_degrees)
    theta: ndarray
        second coordinate in radians (or degrees if use_degrees)
    use_degrees: bool
        if set, takes input angles in degrees and produces outputs also in degrees

    Returns
    -------
    newpos: ndarray
        final positions
    """
    r = np.ones_like(phi)

    # compute new precession constants
    rotMat = get_FK5PrecessMatrix(fromDate, toDate)

    if use_degrees:
        xyz = sphericalToCartesian(r, np.deg2rad(phi), np.deg2rad(theta))
    else:
        xyz = sphericalToCartesian(r, phi, theta)

    tempP = np.array(xyz).T

    xyzrot = np.array([np.dot(rotMat, k) for k in tempP])
    rab = cartesianToSpherical(xyzrot[:, 0], xyzrot[:, 1], xyzrot[:, 2])
    newpos = np.array([rab[1], rab[2]]).T

    if use_degrees:
        return np.rad2deg(newpos)
    else:
        return newpos
