import numpy as np
import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngle
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import (
    EclipseDetector,
    LatitudeCrossingDetector,
    NodeDetector,
)
from org.orekit.propagation.events.handlers import (
    StopOnDecreasing,
    StopOnEvent,
    StopOnIncreasing,
)
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions
from boinor.bodies import Earth
from boinor.twobody import Orbit
from boinor.twobody.events import (
    LatitudeCrossEvent,
    NodeCrossEvent,
    PenumbraEvent,
    UmbraEvent,
)
from boinor.twobody.propagation import cowell

import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Setup orekit virtual machine and associated data
VM = orekit.initVM()
setup_orekit_curdir("orekit-data.zip")

# Macro for conversion between degrees and radians
DEG_TO_RAD = np.pi / 180

# Collect orekit bodies and properties
Sun_orekit, RSun_orekit = CelestialBodyFactory.getSun(), Constants.SUN_RADIUS
# orekit requires that the attractor is of the ~OneAxisEllipsoid so its event
# detector can properly function. Therefore, the Earth is instantiated from this
# class although the flatteing factor is set to zero so it still becomes a
# perfect sphere
REarth_orekit = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
Earth_orekit = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    float(0),
    FramesFactory.getITRF(IERSConventions.IERS_2010, True),
)

# The COE for the orbit to be defined
a, ecc, inc, raan, argp, nu = (6828137.0, 0.0073, 87.0, 20.0, 10.0, 0)

# Define the orkeit and boinor orbits to be used for all the event validation
# cases in this script
epoch0_orekit = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, TimeScalesFactory.getUTC())
ss0_orekit = KeplerianOrbit(
    float(a),
    float(ecc),
    float(inc * DEG_TO_RAD),
    float(argp * DEG_TO_RAD),
    float(raan * DEG_TO_RAD),
    float(nu * DEG_TO_RAD),
    PositionAngle.TRUE,
    FramesFactory.getEME2000(),
    epoch0_orekit,
    Constants.WGS84_EARTH_MU,
)
epoch0_boinor = Time("2020-01-01", scale="utc")
ss0_boinor = Orbit.from_classical(
    Earth,
    a * u.m,
    ecc * u.one,
    inc * u.deg,
    raan * u.deg,
    argp * u.deg,
    nu * u.deg,
    epoch0_boinor,
)

DICT_OF_EVENTS = {
    "umbra-entry": [
        # orekit umbra eclipse detector
        EclipseDetector(Sun_orekit, RSun_orekit, Earth_orekit)
        .withUmbra()
        .withHandler(StopOnDecreasing()),
        # boinor umbra eclipse detector
        UmbraEvent(ss0_boinor, terminal=True, direction=1),
    ],
    "umbra-exit": [
        # orekit umbra eclipse detector
        EclipseDetector(Sun_orekit, RSun_orekit, Earth_orekit)
        .withUmbra()
        .withHandler(StopOnIncreasing()),
        # boinor umbra eclipse detector
        UmbraEvent(ss0_boinor, terminal=True, direction=-1),
    ],
    "penumbra-entry": [
        # orekit penumbra eclipse detector
        EclipseDetector(Sun_orekit, RSun_orekit, Earth_orekit)
        .withPenumbra()
        .withHandler(StopOnDecreasing()),
        # boinor penumbra eclipse detector
        PenumbraEvent(ss0_boinor, terminal=True, direction=1),
    ],
    "penumbra-exit": [
        # orekit penumbra eclipse detector
        EclipseDetector(Sun_orekit, RSun_orekit, Earth_orekit)
        .withPenumbra()
        .withHandler(StopOnIncreasing()),
        # boinor penumbra eclipse detector
        PenumbraEvent(ss0_boinor, terminal=True, direction=-1),
    ],
    "latitude-entry": [
        # orekit latitude crossing event
        LatitudeCrossingDetector(Earth_orekit, 45.00 * DEG_TO_RAD).withHandler(
            StopOnDecreasing()
        ),
        # boinor latitude crossing event
        LatitudeCrossEvent(ss0_boinor, 45.00 * u.deg, terminal=True, direction=-1),
    ],
    "latitude-exit": [
        # orekit latitude crossing event
        LatitudeCrossingDetector(Earth_orekit, 45.00 * DEG_TO_RAD).withHandler(
            StopOnIncreasing()
        ),
        # boinor latitude crossing event
        LatitudeCrossEvent(ss0_boinor, 45.00 * u.deg, terminal=True, direction=1),
    ],
    "node-cross": [
        NodeDetector(ss0_orekit, ss0_orekit.frame).withHandler(StopOnEvent()),
        NodeCrossEvent(terminal=True),
    ],
}
"""A dictionary holding the orekitEvent, the boinorEvent and the absolute and
relative tolerances for the assertion test."""


@pytest.mark.parametrize("event_name", DICT_OF_EVENTS.keys())
def validate_event_detector(event_name):

    # Unpack orekit and boinor events
    orekit_event, boinor_event = DICT_OF_EVENTS[event_name]

    # Time of fliht for propagating the orbit
    tof = float(2 * 24 * 3600)

    # Build the orekit propagator and add the event to it.
    propagator = KeplerianPropagator(ss0_orekit)
    propagator.addEventDetector(orekit_event)

    # Propagate orekit's orbit
    state = propagator.propagate(epoch0_orekit, epoch0_orekit.shiftedBy(tof))
    orekit_event_epoch_raw = state.orbit.getPVCoordinates().getDate()
    # Convert orekit epoch to astropy Time instance
    orekit_event_epoch_str = orekit_event_epoch_raw.toString(TimeScalesFactory.getUTC())
    orekit_event_epoch = Time(orekit_event_epoch_str, scale="utc", format="isot")
    orekit_event_epoch.format = "iso"
    print(f"{orekit_event_epoch}")

    # Propagate boinor's orbit
    _, _ = cowell(
        Earth.k,
        ss0_boinor.r,
        ss0_boinor.v,
        # Generate a set of tofs, each one for each propagation second
        np.linspace(0, tof, 100) * u.s,
        events=[boinor_event],
    )
    boinor_event_epoch = ss0_boinor.epoch + boinor_event.last_t
    print(f"{boinor_event_epoch}")

    # Test both event epochs by checking the distance in seconds between them
    dt = np.abs((orekit_event_epoch - boinor_event_epoch).to(u.s))
    assert_quantity_allclose(dt, 0 * u.s, atol=5 * u.s, rtol=1e-7)
