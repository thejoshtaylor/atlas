"""Wave-0 scaffolds for `CameraAudioSource` (VOICE-04, SRC-02, SRC-04, PROV-07).

Red on purpose -- `src/spire_voice/transports/camera.py` does not exist yet.
Each test below names the plan that turns it green and the exact behavior
it will assert once that plan lands, matching the shape plans 01-01 and
01.1-01 both used for their own Wave-0 scaffolds.
"""

from __future__ import annotations


def test_camera_source_satisfies_audio_source_end_to_end():
    """SRC-02: `CameraAudioSource` must satisfy `AudioSource` against fake
    RTSP packet bytes, the same way `tests/test_transports.py` proves the
    two browser transports do -- turned green by plan 02-03.
    """
    raise AssertionError("plan 02-03 turns this green: CameraAudioSource does not exist yet")


def test_camera_source_yields_bit_identical_alaw_no_transcode():
    """VOICE-04 / PROV-07: the bytes `frames()` yields for the camera source
    must be bit-identical to the packet's own undecoded A-law bitstream --
    no decode, no re-encode, anywhere on the STT-bound path -- turned green
    by plan 02-03.
    """
    raise AssertionError("plan 02-03 turns this green: CameraAudioSource does not exist yet")


def test_camera_source_reconnects_after_a_dropped_connection():
    """SRC-04: a dropped RTSP connection must reconnect on its own -- neither
    PyAV nor FFmpeg do this automatically. Deliberately left red by plan
    02-03 (which owns the two tests above); turned green by plan 02-07,
    which adds the supervised reconnect loop.
    """
    raise AssertionError("plan 02-07 turns this green: the reconnect supervisor does not exist yet")
