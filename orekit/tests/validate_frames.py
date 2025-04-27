""" Validate boinor frames against Orekit ones """

from itertools import product

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.tests.helper import assert_quantity_allclose
from numpy.linalg import norm
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.frames import Frame, FramesFactory, Transform
from org.orekit.time import AbsoluteDate
from org.orekit.utils import (
    IERSConventions,
    PVCoordinatesProvider,
    TimeStampedPVCoordinates,
)
from boinor.bodies import Earth as Earth_boinor
from boinor.bodies import Jupiter as Jupiter_boinor
from boinor.bodies import Mars as Mars_boinor
from boinor.bodies import Mercury as Mercury_boinor
from boinor.bodies import Moon as Moon_boinor
from boinor.bodies import Neptune as Neptune_boinor
from boinor.bodies import Saturn as Saturn_boinor
from boinor.bodies import Sun as Sun_boinor
from boinor.bodies import Uranus as Uranus_boinor
from boinor.bodies import Venus as Venus_boinor
from boinor.constants import J2000 as J2000_BOINOR
from boinor.frames.equatorial import GCRS as GCRS_boinor
from boinor.frames.equatorial import HCRS as HCRS_boinor
from boinor.frames.equatorial import JupiterICRS as JupiterICRS_boinor
from boinor.frames.equatorial import MarsICRS as MarsICRS_boinor
from boinor.frames.equatorial import MercuryICRS as MercuryICRS_boinor
from boinor.frames.equatorial import MoonICRS as MoonICRS_boinor
from boinor.frames.equatorial import NeptuneICRS as NeptuneICRS_boinor
from boinor.frames.equatorial import SaturnICRS as SaturnICRS_boinor
from boinor.frames.equatorial import UranusICRS as UranusICRS_boinor
from boinor.frames.equatorial import VenusICRS as VenusICRS_boinor
from boinor.frames.fixed import ITRS as ITRS_boinor
from boinor.frames.fixed import JupiterFixed as JupiterFixed_boinor
from boinor.frames.fixed import MarsFixed as MarsFixed_boinor
from boinor.frames.fixed import MercuryFixed as MercuryFixed_boinor
from boinor.frames.fixed import MoonFixed as MoonFixed_boinor
from boinor.frames.fixed import NeptuneFixed as NeptuneFixed_boinor
from boinor.frames.fixed import SaturnFixed as SaturnFixed_boinor
from boinor.frames.fixed import SunFixed as SunFixed_boinor
from boinor.frames.fixed import UranusFixed as UranusFixed_boinor
from boinor.frames.fixed import VenusFixed as VenusFixed_boinor

import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Setup orekit virtual machine and associated data
VM = orekit.initVM()
setup_orekit_curdir("orekit-data.zip")

# All interesting 3D directions
R_SET = [
    vector for vector in product([-1, 0, 1], repeat=3) if list(vector) != [0, 0, 0]
]
V_SET = R_SET.copy()

# Retrieve celestial bodies from orekit
Sun_orekit = CelestialBodyFactory.getSun()
Mercury_orekit = CelestialBodyFactory.getMercury()
Venus_orekit = CelestialBodyFactory.getVenus()
Earth_orekit = CelestialBodyFactory.getEarth()
Moon_orekit = CelestialBodyFactory.getMoon()
Mars_orekit = CelestialBodyFactory.getMars()
Jupiter_orekit = CelestialBodyFactory.getJupiter()
Saturn_orekit = CelestialBodyFactory.getSaturn()
Uranus_orekit = CelestialBodyFactory.getUranus()
Neptune_orekit = CelestialBodyFactory.getNeptune()

# Name of the bodies
BODIES_NAMES = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Moon",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]

# orekit: bodies, inertial and fixed frames
OREKIT_BODIES = [
    Sun_orekit,
    Mercury_orekit,
    Venus_orekit,
    Earth_orekit,
    Moon_orekit,
    Mars_orekit,
    Jupiter_orekit,
    Saturn_orekit,
    Uranus_orekit,
    Neptune_orekit,
]
OREKIT_FIXED_FRAMES = [body.getBodyOrientedFrame() for body in OREKIT_BODIES]

# boinor: bodies, inertial and fixed frames
BOINOR_BODIES = [
    Sun_boinor,
    Mercury_boinor,
    Venus_boinor,
    Earth_boinor,
    Moon_boinor,
    Mars_boinor,
    Jupiter_boinor,
    Saturn_boinor,
    Uranus_boinor,
    Neptune_boinor,
]
BOINOR_ICRS_FRAMES = [
    HCRS_boinor,
    MercuryICRS_boinor,
    VenusICRS_boinor,
    GCRS_boinor,
    MoonICRS_boinor,
    MarsICRS_boinor,
    JupiterICRS_boinor,
    SaturnICRS_boinor,
    UranusICRS_boinor,
    NeptuneICRS_boinor,
]
BOINOR_FIXED_FRAMES = [
    SunFixed_boinor,
    MercuryFixed_boinor,
    VenusFixed_boinor,
    ITRS_boinor,
    MoonFixed_boinor,
    MarsFixed_boinor,
    JupiterFixed_boinor,
    SaturnFixed_boinor,
    UranusFixed_boinor,
    NeptuneFixed_boinor,
]


# Collect both API data in two dictionaries
OREKIT_BODIES_AND_FRAMES = dict(
    zip(BODIES_NAMES, zip(OREKIT_BODIES, OREKIT_FIXED_FRAMES))
)
BOINOR_BODIES_AND_FRAMES = dict(
    zip(
        BODIES_NAMES,
        zip(BOINOR_BODIES, BOINOR_ICRS_FRAMES, BOINOR_FIXED_FRAMES),
    )
)


# Macros for J2000 and ICRF frame from Orekit API
J2000_OREKIT = AbsoluteDate.J2000_EPOCH
ICRF_FRAME_OREKIT = FramesFactory.getICRF()
GCRF_FRAME_OREKIT = FramesFactory.getGCRF()
ITRF_FRAME_OREKIT = FramesFactory.getITRF(IERSConventions.IERS_2010, False)


# Some of tests are marked as XFAIL since Orekit implements the data from IAU
# WGCCRE 2009 report while boinor uses IAU WGCCRE 2015 one


@pytest.mark.parametrize("r_vec", R_SET)
@pytest.mark.parametrize("v_vec", V_SET)
@pytest.mark.parametrize(
    "body_name",
    [
        "Sun",
        pytest.param(
            "Mercury", marks=pytest.mark.xfail
        ),  # boinor WGCCRE 2015 report != orekit IAU 2009 report
        "Venus",
        "Moon",
        pytest.param(
            "Mars", marks=pytest.mark.xfail
        ),  # boinor WGCCRE 2015 report != orekit WGCCRE 2009 report
        "Jupiter",
        "Saturn",
        "Uranus",
        pytest.param(
            "Neptune", marks=pytest.mark.xfail
        ),  # boinor WGCCRE 2015 report != orekit IAU 2009 report
    ],
)
def validate_from_body_intertial_to_body_fixed(body_name, r_vec, v_vec):

    # boinor: collect body information
    (
        BODY_BOINOR,
        BODY_ICRF_FRAME_BOINOR,
        BODY_FIXED_FRAME_BOINOR,
    ) = BOINOR_BODIES_AND_FRAMES[body_name]

    # Compute for the norm of position and velocity vectors
    r_norm, v_norm = [norm(vec) for vec in [r_vec, v_vec]]
    R = BODY_BOINOR.R.to(u.m).value

    # Make a position vector who's norm is equal to the body's radius. Make a
    # unitary velocity vector. Units are in [m] and [m / s].
    rx, ry, rz = [float(r_i * R / r_norm) for r_i in r_vec]
    vx, vy, vz = [float(v_i / v_norm) for v_i in v_vec]

    # boinor: build r_vec and v_vec wrt inertial body frame
    xyz_boinor = CartesianRepresentation(rx * u.m, ry * u.m, rz * u.m)
    coords_wrt_bodyICRS_boinor = BODY_ICRF_FRAME_BOINOR(xyz_boinor)

    # boinor: convert from inertial to fixed frame at given epoch
    coords_wrt_bodyFIXED_boinor = (
        coords_wrt_bodyICRS_boinor.transform_to(
            BODY_FIXED_FRAME_BOINOR(obstime=J2000_BOINOR)
        )
        .represent_as(CartesianRepresentation)
        .xyz
    )

    # orekit: collect body information
    (BODY_OREKIT, BODY_FIXED_FRAME_OREKIT,) = OREKIT_BODIES_AND_FRAMES[body_name]

    # orekit: build r_vec and v_vec wrt inertial body frame
    xyz_orekit = Vector3D(rx, ry, rz)
    uvw_orekit = Vector3D(vx, vy, vz)
    coords_wrt_bodyICRS_orekit = TimeStampedPVCoordinates(
        J2000_OREKIT, xyz_orekit, uvw_orekit
    )

    # orekit: create bodyICRS frame as pure translation of ICRF one
    coords_body_wrt_ICRF_orekit = PVCoordinatesProvider.cast_(
        BODY_OREKIT
    ).getPVCoordinates(J2000_OREKIT, ICRF_FRAME_OREKIT)
    BODY_ICRF_FRAME_OREKIT = Frame(
        ICRF_FRAME_OREKIT,
        Transform(J2000_OREKIT, coords_body_wrt_ICRF_orekit.negate()),
        body_name.capitalize() + "ICRF",
    )

    # orekit: build conversion between BodyICRF and BodyFixed frames
    bodyICRF_to_bodyFIXED_orekit = BODY_ICRF_FRAME_OREKIT.getTransformTo(
        BODY_FIXED_FRAME_OREKIT, J2000_OREKIT,
    )

    # orekit: convert from inertial coordinates to non-inertial ones
    coords_orekit_fixed_raw = (
        bodyICRF_to_bodyFIXED_orekit.transformPVCoordinates(coords_wrt_bodyICRS_orekit)
        .getPosition()
        .toArray()
    )
    coords_wrt_bodyFIXED_orekit = np.asarray(coords_orekit_fixed_raw) * u.m

    # Check position conversion
    assert_quantity_allclose(
        coords_wrt_bodyFIXED_boinor,
        coords_wrt_bodyFIXED_orekit,
        atol=1e-5 * u.m,
        rtol=1e-7,
    )


@pytest.mark.parametrize("r_vec", R_SET)
@pytest.mark.parametrize("v_vec", V_SET)
def validate_GCRF_to_ITRF(r_vec, v_vec):

    # Compute for the norm of position and velocity vectors
    r_norm, v_norm = [norm(vec) for vec in [r_vec, v_vec]]
    R = Earth_boinor.R.to(u.m).value

    # Correction factor to normalize position and velocity vectors
    k_r = R / r_norm if r_norm != 0 else 1.00
    k_v = 1 / v_norm if v_norm != 0 else 1.00

    # Make a position vector who's norm is equal to the body's radius. Make a
    # unitary velocity vector. Units are in [m] and [m / s].
    rx, ry, rz = [float(k_r * r_i) for r_i in r_vec]
    vx, vy, vz = [float(k_v * v_i) for v_i in v_vec]

    # orekit: build r_vec and v_vec wrt inertial body frame
    xyz_orekit = Vector3D(rx, ry, rz)
    uvw_orekit = Vector3D(vx, vy, vz)
    coords_GCRF_orekit = TimeStampedPVCoordinates(J2000_OREKIT, xyz_orekit, uvw_orekit)

    # orekit: build conversion between GCRF and ITRF
    GCRF_TO_ITRF_OREKIT = GCRF_FRAME_OREKIT.getTransformTo(
        ITRF_FRAME_OREKIT, J2000_OREKIT,
    )

    # orekit: convert from GCRF to ITRF using previous built conversion
    coords_ITRF_orekit = (
        GCRF_TO_ITRF_OREKIT.transformPVCoordinates(coords_GCRF_orekit)
        .getPosition()
        .toArray()
    )
    coords_ITRF_orekit = np.asarray(coords_ITRF_orekit) * u.m

    # boinor: build r_vec and v_vec wrt GCRF
    xyz_boinor = CartesianRepresentation(rx * u.m, ry * u.m, rz * u.m)
    coords_GCRS_boinor = GCRS_boinor(xyz_boinor)

    # boinor: convert from inertial to fixed frame at given epoch
    coords_ITRS_boinor = (
        coords_GCRS_boinor.transform_to(ITRS_boinor(obstime=J2000_BOINOR))
        .represent_as(CartesianRepresentation)
        .xyz
    )

    # Check position conversion
    assert_quantity_allclose(
        coords_ITRS_boinor, coords_ITRF_orekit, atol=1e-3 * u.m, rtol=1e-2,
    )
