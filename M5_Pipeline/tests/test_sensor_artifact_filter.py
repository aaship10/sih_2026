"""Hand-computed cases for planner.py's sensor-artifact filters -- the fix
for a confirmed live-CARLA-testing incident (see config.py's matching
comment): a persistent class='unknown', LiDAR-only-confidence, near-zero-
velocity track next to the ego froze it in EMERGENCY_BRAKE, which triggered
scenario3.py's stuck-recovery teleport, which caused a real collision. The
two actual offending tracks were M3-flagged is_static=True (StaticObstacle,
no velocity field) -- confirmed by replaying the exact recorded incident
data, which is why both the dynamic (_is_likely_sensor_artifact) and static
(_is_likely_sensor_artifact_static) filters are covered here.

A SECOND live incident (same track_id, confirmed via M3's own is_static
flag flipping False on it mid-run) found the same ghost's reported WORLD
velocity climbing smoothly toward the ego's own as the ego drove away --
the exact "noise misread as motion" pattern M4's own multimodal/features.py
already documents and fixes for its OWN prediction math, never applied to
the raw velocity M4 forwards to M5. Checking absolute world-frame speed
missed this entirely (a ghost tracking alongside a moving vehicle reports
nontrivial world velocity); _is_likely_sensor_artifact now checks velocity
RELATIVE TO THE EGO instead -- test_ghost_matching_ego_velocity_is_still_
filtered below is that exact regression case.

Replaying the ACTUAL recorded second incident then found even THAT wasn't
enough: the ghost's velocity reading turned out to be independently
unreliable (climbing then decaying over 40+ frames with no relationship to
the ego's real motion, confirmed by it staying elevated through 20+ frames
where the ego was completely stationary) -- neither absolute nor relative
speed ever dropped below threshold during the window this ghost was
active. The signal that DOES hold throughout is its DISTANCE to the ego
staying in a narrow band the whole time -- _distance_is_stable and the
distance_history-based path in _is_likely_sensor_artifact are that fix;
test_stability_path_catches_what_velocity_alone_missed below reproduces
the exact incident numbers.

Every "must NOT be filtered" case below is the actual safety property this
filter must preserve: it should only ever remove the EXACT signature that
caused the incident, never a genuine close call."""
import config
from planner import _distance_is_stable, _is_likely_sensor_artifact, _is_likely_sensor_artifact_static
from schema import Obstacle, StaticObstacle


def _obstacle(class_name="unknown", confidence=0.4, pos=(2.0, 0.0), vel=(0.0, 0.0)):
    return Obstacle(
        track_id=1, class_name=class_name, confidence=confidence,
        pos_x=pos[0], pos_y=pos[1], vel_x=vel[0], vel_y=vel[1],
        width=1.0, length=1.0, trajectories=[],
    )


def _static(class_name="unknown", confidence=0.4, pos=(2.0, 0.0)):
    return StaticObstacle(
        track_id=1, class_name=class_name, confidence=confidence,
        pos_x=pos[0], pos_y=pos[1], width=1.0, length=1.0,
    )


def test_the_exact_observed_incident_is_filtered():
    # track 23 from the live incident: unknown, confidence 0.40, ~1.95m away,
    # near-zero velocity, ego stationary -- relative speed == absolute speed here.
    obs = _obstacle(confidence=0.40, pos=(1.95, 0.0), vel=(-0.016, 0.063))
    assert _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)


def test_ghost_matching_ego_velocity_is_still_filtered():
    # The SECOND live incident this fix specifically targets: the ghost's
    # ABSOLUTE world velocity (2.9 m/s) climbed toward the ego's own speed
    # as the ego drove away (2.9 m/s here too) -- an absolute-speed check
    # would wrongly let this through; relative-to-ego speed (~0) correctly
    # still flags it.
    obs = _obstacle(confidence=0.40, pos=(1.5, 2.0), vel=(0.02, 2.90))
    assert _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=2.90)


def test_the_exact_observed_incident_is_filtered_as_static():
    # the incident's actual tracks (22 and 23) were both is_static=True.
    for pos in [(1.95, 0.0), (3.55, 0.0)]:
        assert _is_likely_sensor_artifact_static(_static(pos=pos), ego_x=0.0, ego_y=0.0)


def test_known_class_close_static_is_never_filtered():
    obs = _static(class_name="car", pos=(1.0, 0.0))
    assert not _is_likely_sensor_artifact_static(obs, ego_x=0.0, ego_y=0.0)


def test_higher_confidence_close_static_is_never_filtered():
    obs = _static(confidence=0.6, pos=(1.0, 0.0))
    assert not _is_likely_sensor_artifact_static(obs, ego_x=0.0, ego_y=0.0)


def test_far_away_low_confidence_static_is_never_filtered():
    obs = _static(pos=(50.0, 0.0))
    assert not _is_likely_sensor_artifact_static(obs, ego_x=0.0, ego_y=0.0)


def test_known_class_close_object_is_never_filtered():
    obs = _obstacle(class_name="pedestrian", confidence=0.4, pos=(1.0, 0.0), vel=(0.0, 0.0))
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)


def test_higher_confidence_unknown_object_is_never_filtered():
    # confidence > 0.4 means camera corroboration exists -- not a LiDAR-only ghost.
    obs = _obstacle(class_name="unknown", confidence=0.6, pos=(1.0, 0.0), vel=(0.0, 0.0))
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)


def test_far_away_unknown_low_confidence_object_is_never_filtered():
    obs = _obstacle(pos=(50.0, 0.0), vel=(0.0, 0.0))
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)


def test_fast_moving_unknown_low_confidence_object_is_never_filtered():
    # a real closing hazard, even if unclassified and low-confidence, must
    # never be filtered -- moving fast RELATIVE TO A STATIONARY EGO.
    obs = _obstacle(pos=(2.0, 0.0), vel=(5.0, 0.0))
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)


def test_object_moving_relative_to_a_moving_ego_is_never_filtered():
    # A real hazard closing on a MOVING ego (not matching its velocity) --
    # the relative-velocity check must still catch this, not just the
    # stationary-ego case above.
    obs = _obstacle(pos=(2.0, 0.0), vel=(5.0, 0.0))
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=5.0, ego_vy=5.0)


def test_boundary_distance_is_inclusive():
    obs = _obstacle(pos=(config.SENSOR_ARTIFACT_EXCLUSION_RADIUS_M, 0.0))
    assert _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)
    obs_far = _obstacle(pos=(config.SENSOR_ARTIFACT_EXCLUSION_RADIUS_M + 0.01, 0.0))
    assert not _is_likely_sensor_artifact(obs_far, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0)


def test_distance_is_stable_requires_minimum_samples():
    short_history = [2.0] * (config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES - 1)
    assert not _distance_is_stable(short_history, config.SENSOR_ARTIFACT_STABILITY_BAND_M, config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES)


def test_distance_is_stable_true_within_band():
    # the live incident's actual observed range: 1.2m to 2.7m (a 1.5m band).
    history = [1.2, 1.5, 1.83, 2.21, 2.53, 2.69, 2.53, 2.21]
    assert len(history) == config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES
    assert _distance_is_stable(history, config.SENSOR_ARTIFACT_STABILITY_BAND_M, config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES)


def test_distance_is_stable_false_outside_band():
    # a real object closing distance from 10m to 2m over the window -- NOT stable.
    history = [10.0, 8.5, 7.0, 5.5, 4.0, 3.0, 2.5, 2.0]
    assert not _distance_is_stable(history, config.SENSOR_ARTIFACT_STABILITY_BAND_M, config.SENSOR_ARTIFACT_STABILITY_MIN_SAMPLES)


def test_stability_path_catches_what_velocity_alone_missed():
    # Reproduces the actual recorded second-incident numbers at the frame
    # where the earlier (velocity-only) fix failed: relative speed ~3.0
    # m/s (both absolute AND relative-to-ego exceed the 0.5 m/s threshold),
    # but distance-to-ego history stayed within a 1.5m band.
    obs = _obstacle(confidence=0.40, pos=(2.23, 0.0), vel=(-0.026, 0.811))
    history = [1.2, 1.5, 1.83, 2.21, 2.53, 2.69, 2.53, 2.21]
    # Without history: not caught (this is the exact gap the previous fix left).
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.033, ego_vy=3.80)
    # With the stability history: caught.
    assert _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.033, ego_vy=3.80, distance_history=history)


def test_stability_history_never_masks_a_genuinely_unstable_real_object():
    # A real object whose distance IS varying a lot must never be filtered,
    # even though it happens to be class='unknown'/low-confidence and
    # briefly reads a high relative speed at this instant.
    obs = _obstacle(confidence=0.40, pos=(2.0, 0.0), vel=(5.0, 0.0))
    unstable_history = [10.0, 8.5, 7.0, 5.5, 4.0, 3.0, 2.5, 2.0]
    assert not _is_likely_sensor_artifact(obs, ego_x=0.0, ego_y=0.0, ego_vx=0.0, ego_vy=0.0, distance_history=unstable_history)
