""" Validate boinor's two-body element conversion against Orekit"""

import numpy as np
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngle
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants as C
from org.orekit.utils import PVCoordinates
from boinor.bodies import Earth
from boinor.twobody import Orbit

import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Setup orekit virtual machine and associated data
VM = orekit.initVM()
setup_orekit_curdir("orekit-data.zip")


def validate_rv2coe():
    # Orbit initial state vectors
    rx_0, ry_0, rz_0 = -6045.0e3, -3490.0e3, 2500.0e3
    vx_0, vy_0, vz_0 = -3.457e3, 6.618e3, 2.533e3

    # Build the initial Orekit orbit
    k = C.IAU_2015_NOMINAL_EARTH_GM
    r0_vec = Vector3D(rx_0, ry_0, rz_0)
    v0_vec = Vector3D(vx_0, vy_0, vz_0)
    rv_0 = PVCoordinates(r0_vec, v0_vec)
    epoch_00 = AbsoluteDate.J2000_EPOCH
    gcrf_frame = FramesFactory.getGCRF()
    ss0_orekit = KeplerianOrbit(rv_0, gcrf_frame, epoch_00, k)

    # Build boinor orbit
    r0_vec = [rx_0, ry_0, rz_0] * u.m
    v0_vec = [vx_0, vy_0, vz_0] * u.m / u.s
    ss0_boinor = Orbit.from_vectors(Earth, r0_vec, v0_vec)

    # Orekit bounds COE angular magnitudes between [-pi, +pi]. Let us define a
    # converter function to map values between [0, +2pi]
    def unbound_angle(angle):
        """ Bound angle between [0, +2pi] """
        if -np.pi <= angle <= 0.0:
            angle += 2 * np.pi

        return angle

    # Map angles
    orekit_raan = unbound_angle(ss0_orekit.getRightAscensionOfAscendingNode())
    orekit_argp = unbound_angle(ss0_orekit.getPerigeeArgument())
    orekit_nu = unbound_angle(ss0_orekit.getTrueAnomaly())

    # Assert classical orbital elements
    assert_quantity_allclose(ss0_boinor.a, ss0_orekit.getA() * u.m, rtol=1e-6)
    assert_quantity_allclose(ss0_boinor.ecc, ss0_orekit.getE() * u.one, rtol=1e-6)
    assert_quantity_allclose(ss0_boinor.inc, ss0_orekit.getI() * u.rad, rtol=1e-6)
    assert_quantity_allclose(ss0_boinor.raan, orekit_raan * u.rad, rtol=1e-6)
    assert_quantity_allclose(ss0_boinor.argp, orekit_argp * u.rad, rtol=1e-6)
    assert_quantity_allclose(ss0_boinor.nu, orekit_nu * u.rad, rtol=1e-6)


def validate_coe2rv():

    # Initial COE of the orbit
    a, ecc, inc, raan, argp, nu = 8788e3, 0.1712, 2.6738, 4.4558, 0.3502, 0.4965

    # Build Orekit orbit
    k = C.IAU_2015_NOMINAL_EARTH_GM
    epoch_00 = AbsoluteDate.J2000_EPOCH
    gcrf_frame = FramesFactory.getGCRF()
    ss0_orekit = KeplerianOrbit(
        a, ecc, inc, argp, raan, nu, PositionAngle.TRUE, gcrf_frame, epoch_00, k
    )

    # Build boinor orbit
    ss0_boinor = Orbit.from_classical(
        Earth,
        a * u.m,
        ecc * u.one,
        inc * u.rad,
        raan * u.rad,
        argp * u.rad,
        nu * u.rad,
    )

    # Retrieve Orekit final state vectors
    r_orekit = ss0_orekit.getPVCoordinates().position.toArray() * u.m
    v_orekit = ss0_orekit.getPVCoordinates().velocity.toArray() * u.m / u.s

    # Retrieve boinor final state vectors
    r_boinor, v_boinor = ss0_boinor.rv()

    # Assert final state vector
    assert_quantity_allclose(r_boinor, r_orekit)
    assert_quantity_allclose(v_boinor, v_orekit)
