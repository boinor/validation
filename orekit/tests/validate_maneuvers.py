""" Validate boinor impulsive maneuvers against Orekit """

import numpy as np
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.attitudes import LofOffset
from org.orekit.forces.maneuvers import ImpulseManeuver
from org.orekit.frames import FramesFactory, LOFType
from org.orekit.orbits import KeplerianOrbit
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import ApsideDetector
from org.orekit.propagation.events.handlers import StopOnDecreasing
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants as C
from org.orekit.utils import PVCoordinates
from boinor.bodies import Earth
from boinor.maneuver import Maneuver
from boinor.twobody import Orbit

import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Setup orekit virtual machine and associated data
VM = orekit.initVM()
setup_orekit_curdir("orekit-data.zip")


def validate_3D_hohmann():

    # Initial orbit state vectors, final radius and time of flight
    # This orbit has an inclination of 22.30 [deg] so the maneuver will not be
    # applied on an equatorial orbit. See associated GMAT script:
    # gmat_validate_hohmann3D.script
    rx_0, ry_0, rz_0 = 7200e3, -1000.0e3, 0.0
    vx_0, vy_0, vz_0 = 0.0, 8.0e3, 3.25e3
    rf_norm = 35781.35e3
    tof = 19800.00

    # Build the initial Orekit orbit
    k = C.IAU_2015_NOMINAL_EARTH_GM
    r0_vec = Vector3D(rx_0, ry_0, rz_0)
    v0_vec = Vector3D(vx_0, vy_0, vz_0)
    rv_0 = PVCoordinates(r0_vec, v0_vec)
    epoch_00 = AbsoluteDate.J2000_EPOCH
    gcrf_frame = FramesFactory.getGCRF()
    ss_0 = KeplerianOrbit(rv_0, gcrf_frame, epoch_00, k)

    # Final desired orbit radius and auxiliary variables
    a_0, ecc_0 = ss_0.getA(), ss_0.getE()
    rp_norm = a_0 * (1 - ecc_0)
    vp_norm = np.sqrt(2 * k / rp_norm - k / a_0)
    a_trans = (rp_norm + rf_norm) / 2

    # Compute the magnitude of Hohmann's deltaV
    dv_a = np.sqrt(2 * k / rp_norm - k / a_trans) - vp_norm
    dv_b = np.sqrt(k / rf_norm) - np.sqrt(2 * k / rf_norm - k / a_trans)

    # Convert previous magnitudes into vectors within the TNW frame, which
    # is aligned with local velocity vector
    dVa_vec, dVb_vec = [Vector3D(float(dV), float(0), float(0)) for dV in (dv_a, dv_b)]

    # Local orbit frame: X-axis aligned with velocity, Z-axis towards momentum
    attitude_provider = LofOffset(gcrf_frame, LOFType.TNW)

    # Setup impulse triggers; default apside detector stops on increasing g
    # function (aka at perigee)
    at_apoapsis = ApsideDetector(ss_0).withHandler(StopOnDecreasing())
    at_periapsis = ApsideDetector(ss_0)

    # Build the impulsive maneuvers; ISP is only related to fuel mass cost
    ISP = float(300)
    imp_A = ImpulseManeuver(at_periapsis, attitude_provider, dVa_vec, ISP)
    imp_B = ImpulseManeuver(at_apoapsis, attitude_provider, dVb_vec, ISP)

    # Generate the propagator and add the maneuvers
    propagator = KeplerianPropagator(ss_0, attitude_provider)
    propagator.addEventDetector(imp_A)
    propagator.addEventDetector(imp_B)

    # Apply the maneuver by propagating the orbit
    epoch_ff = epoch_00.shiftedBy(tof)
    rv_f = propagator.propagate(epoch_ff).getPVCoordinates(gcrf_frame)

    # Retrieve orekit final state vectors
    r_orekit, v_orekit = (
        rv_f.getPosition().toArray() * u.m,
        rv_f.getVelocity().toArray() * u.m / u.s,
    )

    # Build initial boinor orbit and apply the maneuver
    r0_vec = [rx_0, ry_0, rz_0] * u.m
    v0_vec = [vx_0, vy_0, vz_0] * u.m / u.s
    ss0_boinor = Orbit.from_vectors(Earth, r0_vec, v0_vec)
    man_boinor = Maneuver.hohmann(ss0_boinor, rf_norm * u.m)

    # Retrieve propagation time after maneuver has been applied
    tof_prop = tof - man_boinor.get_total_time().to(u.s).value
    ssf_boinor = ss0_boinor.apply_maneuver(man_boinor).propagate(tof_prop * u.s)

    # Retrieve boinor final state vectors
    r_boinor, v_boinor = ssf_boinor.rv()

    # Assert final state vectors
    assert_quantity_allclose(r_boinor, r_orekit, rtol=1e-6)
    assert_quantity_allclose(v_boinor, v_orekit, rtol=1e-6)


def validate_3D_bielliptic():

    # Initial orbit state vectors, final radius and time of flight
    # This orbit has an inclination of 22.30 [deg] so the maneuver will not be
    # applied on an equatorial orbit. See associated GMAT script:
    # gmat_validate_bielliptic3D.script
    rx_0, ry_0, rz_0 = 7200e3, -1000.0e3, 0.0
    vx_0, vy_0, vz_0 = 0.0, 8.0e3, 3.25e3
    rb_norm = 50000.00e3
    rf_norm = 35781.35e3
    tof_trans = 24500.00  # Impulse dVb is performed just a bit before this epoch
    tof = 72158.11

    # Build the initial Orekit orbit
    k = C.IAU_2015_NOMINAL_EARTH_GM
    r0_vec = Vector3D(rx_0, ry_0, rz_0)
    v0_vec = Vector3D(vx_0, vy_0, vz_0)
    rv_0 = PVCoordinates(r0_vec, v0_vec)
    epoch_00 = AbsoluteDate.J2000_EPOCH
    gcrf_frame = FramesFactory.getGCRF()
    ss_0 = KeplerianOrbit(rv_0, gcrf_frame, epoch_00, k)

    # Final desired orbit radius and auxiliary variables
    a_0, ecc_0 = ss_0.getA(), ss_0.getE()
    rp_norm = a_0 * (1 - ecc_0)
    vp_norm = np.sqrt(2 * k / rp_norm - k / a_0)
    a_trans1 = (rp_norm + rb_norm) / 2
    a_trans2 = (rb_norm + rf_norm) / 2

    # Compute the magnitude of Bielliptic's deltaV
    dv_a = np.sqrt(2 * k / rp_norm - k / a_trans1) - vp_norm
    dv_b = np.sqrt(2 * k / rb_norm - k / a_trans2) - np.sqrt(
        2 * k / rb_norm - k / a_trans1
    )
    dv_c = np.sqrt(k / rf_norm) - np.sqrt(2 * k / rf_norm - k / a_trans2)

    # Convert previous magnitudes into vectors within the TNW frame, which
    # is aligned with local velocity vector
    dVa_vec, dVb_vec, dVc_vec = [
        Vector3D(float(dV), float(0), float(0)) for dV in (dv_a, dv_b, dv_c)
    ]

    # Local orbit frame: X-axis aligned with velocity, Z-axis towards momentum
    attitude_provider = LofOffset(gcrf_frame, LOFType.TNW)

    # Setup impulse triggers; default apside detector stops on increasing g
    # function (aka at perigee)
    at_apoapsis = ApsideDetector(ss_0).withHandler(StopOnDecreasing())
    at_periapsis = ApsideDetector(ss_0)

    # Build the impulsive maneuvers; ISP is only related to fuel mass cost
    ISP = float(300)
    imp_A = ImpulseManeuver(at_periapsis, attitude_provider, dVa_vec, ISP)
    imp_B = ImpulseManeuver(at_apoapsis, attitude_provider, dVb_vec, ISP)

    # Generate the propagator and add the first two impulses
    propagator = KeplerianPropagator(ss_0, attitude_provider)
    propagator.addEventDetector(imp_A)
    propagator.addEventDetector(imp_B)

    # Apply the maneuver by propagating the orbit
    epoch_trans = epoch_00.shiftedBy(tof_trans)
    ss_trans = propagator.propagate(epoch_trans).getOrbit()

    # Build a second propagator and add the last impulse so it is not confused
    # with the first one as both happen at perigee
    at_periapsis = ApsideDetector(ss_trans)
    imp_C = ImpulseManeuver(at_periapsis, attitude_provider, dVc_vec, ISP)
    propagator = KeplerianPropagator(ss_trans, attitude_provider)
    propagator.addEventDetector(imp_C)

    # Apply the last of the impulses
    tof_ff = float(tof - tof_trans)
    epoch_ff = epoch_trans.shiftedBy(tof_ff)
    rv_f = propagator.propagate(epoch_ff).getPVCoordinates(gcrf_frame)

    # Retrieve orekit final state vectors
    r_orekit, v_orekit = (
        rv_f.getPosition().toArray() * u.m,
        rv_f.getVelocity().toArray() * u.m / u.s,
    )

    # Build initial boinor orbit and apply the maneuver
    r0_vec = [rx_0, ry_0, rz_0] * u.m
    v0_vec = [vx_0, vy_0, vz_0] * u.m / u.s
    ss0_boinor = Orbit.from_vectors(Earth, r0_vec, v0_vec)
    man_boinor = Maneuver.bielliptic(ss0_boinor, rb_norm * u.m, rf_norm * u.m)

    # Retrieve propagation time after maneuver has been applied
    tof_prop = tof - man_boinor.get_total_time().to(u.s).value
    ssf_boinor = ss0_boinor.apply_maneuver(man_boinor).propagate(tof_prop * u.s,)

    # Retrieve boinor final state vectors
    r_boinor, v_boinor = ssf_boinor.rv()

    # Assert final state vectors
    assert_quantity_allclose(r_boinor, r_orekit, rtol=1e-6)
    assert_quantity_allclose(v_boinor, v_orekit, rtol=1e-6)
