""" Validate boinor's two-body propagation against Orekit"""

from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit
from org.orekit.propagation.analytical import KeplerianPropagator
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


def validate_elliptic_propagation():
    # Orbit initial state vectors and time of flight
    rx_0, ry_0, rz_0 = 1131.340e3, -2282.343e3, 6672.423e3
    vx_0, vy_0, vz_0 = -5.64305e3, 4.30333e3, 2.42879e3
    tof = 1.5 * 3600

    # Build the initial Orekit orbit
    k = C.IAU_2015_NOMINAL_EARTH_GM
    r0_vec = Vector3D(rx_0, ry_0, rz_0)
    v0_vec = Vector3D(vx_0, vy_0, vz_0)
    rv_0 = PVCoordinates(r0_vec, v0_vec)
    epoch_00 = AbsoluteDate.J2000_EPOCH
    gcrf_frame = FramesFactory.getGCRF()
    ss0_orekit = KeplerianOrbit(rv_0, gcrf_frame, epoch_00, k)
    orekit_propagator = KeplerianPropagator(ss0_orekit)
    ssf_orekit = orekit_propagator.propagate(epoch_00.shiftedBy(tof)).getOrbit()

    # Build boinor orbit
    r0_vec = [rx_0, ry_0, rz_0] * u.m
    v0_vec = [vx_0, vy_0, vz_0] * u.m / u.s
    ss0_boinor = Orbit.from_vectors(Earth, r0_vec, v0_vec)
    ssf_boinor = ss0_boinor.propagate(tof * u.s)

    # Retrieve Orekit final state vectors
    r_orekit = ssf_orekit.getPVCoordinates().position.toArray() * u.m
    v_orekit = ssf_orekit.getPVCoordinates().velocity.toArray() * u.m / u.s

    # Retrieve boinor final state vectors
    r_boinor, v_boinor = ssf_boinor.rv()

    # Assert final state vectors
    assert_quantity_allclose(r_boinor, r_orekit, atol=10 * u.m)
    assert_quantity_allclose(v_boinor, v_orekit, rtol=1e-5)
