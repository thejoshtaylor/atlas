"""MPEG-TS framing for the Tapo camera's 8800 "talk" endpoint.

A faithful port of go2rtc's `pkg/mpegts/muxer.go`, ported from the proven
prototype at `reference/tapo_talk.py` (run live against a real Tapo C121
this session). The camera's talk endpoint expects exactly this framing: one
PAT packet naming a program, one PMT packet naming a single PCMATapo
(stream type `0x90`) elementary stream, then PES-wrapped A-law payload
packets on that elementary stream's PID.

`tests/test_speaker_mpegts.py` freezes `build_header()`'s and
`get_payload()`'s output against go2rtc's own known-good bytes -- this
module has no pytapo import and no I/O, so that test needs no camera.
"""

from __future__ import annotations

# go2rtc's own CRC-32/MPEG-2 table (reversed polynomial 0xEDB88320,
# byte-at-a-time, reflected), inlined from reference/crc_table.py so this
# module is self-contained.
CRC_TABLE = [
    0x00000000, 0xB71DC104, 0x6E3B8209, 0xD926430D, 0xDC760413, 0x6B6BC517, 0xB24D861A, 0x0550471E,
    0xB8ED0826, 0x0FF0C922, 0xD6D68A2F, 0x61CB4B2B, 0x649B0C35, 0xD386CD31, 0x0AA08E3C, 0xBDBD4F38,
    0x70DB114C, 0xC7C6D048, 0x1EE09345, 0xA9FD5241, 0xACAD155F, 0x1BB0D45B, 0xC2969756, 0x758B5652,
    0xC836196A, 0x7F2BD86E, 0xA60D9B63, 0x11105A67, 0x14401D79, 0xA35DDC7D, 0x7A7B9F70, 0xCD665E74,
    0xE0B62398, 0x57ABE29C, 0x8E8DA191, 0x39906095, 0x3CC0278B, 0x8BDDE68F, 0x52FBA582, 0xE5E66486,
    0x585B2BBE, 0xEF46EABA, 0x3660A9B7, 0x817D68B3, 0x842D2FAD, 0x3330EEA9, 0xEA16ADA4, 0x5D0B6CA0,
    0x906D32D4, 0x2770F3D0, 0xFE56B0DD, 0x494B71D9, 0x4C1B36C7, 0xFB06F7C3, 0x2220B4CE, 0x953D75CA,
    0x28803AF2, 0x9F9DFBF6, 0x46BBB8FB, 0xF1A679FF, 0xF4F63EE1, 0x43EBFFE5, 0x9ACDBCE8, 0x2DD07DEC,
    0x77708634, 0xC06D4730, 0x194B043D, 0xAE56C539, 0xAB068227, 0x1C1B4323, 0xC53D002E, 0x7220C12A,
    0xCF9D8E12, 0x78804F16, 0xA1A60C1B, 0x16BBCD1F, 0x13EB8A01, 0xA4F64B05, 0x7DD00808, 0xCACDC90C,
    0x07AB9778, 0xB0B6567C, 0x69901571, 0xDE8DD475, 0xDBDD936B, 0x6CC0526F, 0xB5E61162, 0x02FBD066,
    0xBF469F5E, 0x085B5E5A, 0xD17D1D57, 0x6660DC53, 0x63309B4D, 0xD42D5A49, 0x0D0B1944, 0xBA16D840,
    0x97C6A5AC, 0x20DB64A8, 0xF9FD27A5, 0x4EE0E6A1, 0x4BB0A1BF, 0xFCAD60BB, 0x258B23B6, 0x9296E2B2,
    0x2F2BAD8A, 0x98366C8E, 0x41102F83, 0xF60DEE87, 0xF35DA999, 0x4440689D, 0x9D662B90, 0x2A7BEA94,
    0xE71DB4E0, 0x500075E4, 0x892636E9, 0x3E3BF7ED, 0x3B6BB0F3, 0x8C7671F7, 0x555032FA, 0xE24DF3FE,
    0x5FF0BCC6, 0xE8ED7DC2, 0x31CB3ECF, 0x86D6FFCB, 0x8386B8D5, 0x349B79D1, 0xEDBD3ADC, 0x5AA0FBD8,
    0xEEE00C69, 0x59FDCD6D, 0x80DB8E60, 0x37C64F64, 0x3296087A, 0x858BC97E, 0x5CAD8A73, 0xEBB04B77,
    0x560D044F, 0xE110C54B, 0x38368646, 0x8F2B4742, 0x8A7B005C, 0x3D66C158, 0xE4408255, 0x535D4351,
    0x9E3B1D25, 0x2926DC21, 0xF0009F2C, 0x471D5E28, 0x424D1936, 0xF550D832, 0x2C769B3F, 0x9B6B5A3B,
    0x26D61503, 0x91CBD407, 0x48ED970A, 0xFFF0560E, 0xFAA01110, 0x4DBDD014, 0x949B9319, 0x2386521D,
    0x0E562FF1, 0xB94BEEF5, 0x606DADF8, 0xD7706CFC, 0xD2202BE2, 0x653DEAE6, 0xBC1BA9EB, 0x0B0668EF,
    0xB6BB27D7, 0x01A6E6D3, 0xD880A5DE, 0x6F9D64DA, 0x6ACD23C4, 0xDDD0E2C0, 0x04F6A1CD, 0xB3EB60C9,
    0x7E8D3EBD, 0xC990FFB9, 0x10B6BCB4, 0xA7AB7DB0, 0xA2FB3AAE, 0x15E6FBAA, 0xCCC0B8A7, 0x7BDD79A3,
    0xC660369B, 0x717DF79F, 0xA85BB492, 0x1F467596, 0x1A163288, 0xAD0BF38C, 0x742DB081, 0xC3307185,
    0x99908A5D, 0x2E8D4B59, 0xF7AB0854, 0x40B6C950, 0x45E68E4E, 0xF2FB4F4A, 0x2BDD0C47, 0x9CC0CD43,
    0x217D827B, 0x9660437F, 0x4F460072, 0xF85BC176, 0xFD0B8668, 0x4A16476C, 0x93300461, 0x242DC565,
    0xE94B9B11, 0x5E565A15, 0x87701918, 0x306DD81C, 0x353D9F02, 0x82205E06, 0x5B061D0B, 0xEC1BDC0F,
    0x51A69337, 0xE6BB5233, 0x3F9D113E, 0x8880D03A, 0x8DD09724, 0x3ACD5620, 0xE3EB152D, 0x54F6D429,
    0x7926A9C5, 0xCE3B68C1, 0x171D2BCC, 0xA000EAC8, 0xA550ADD6, 0x124D6CD2, 0xCB6B2FDF, 0x7C76EEDB,
    0xC1CBA1E3, 0x76D660E7, 0xAFF023EA, 0x18EDE2EE, 0x1DBDA5F0, 0xAAA064F4, 0x738627F9, 0xC49BE6FD,
    0x09FDB889, 0xBEE0798D, 0x67C63A80, 0xD0DBFB84, 0xD58BBC9A, 0x62967D9E, 0xBBB03E93, 0x0CADFF97,
    0xB110B0AF, 0x060D71AB, 0xDF2B32A6, 0x6836F3A2, 0x6D66B4BC, 0xDA7B75B8, 0x035D36B5, 0xB440F7B1,
]

PACKET_SIZE = 188
SYNC = 0x47
PAT_PID = 0
PMT_PID = 0x1000
PES0_PID = 0x100
STREAM_TYPE_PCMA_TAPO = 0x90
STREAM_ID_AUDIO = 0xC0


def crc32_mpegts(data: bytes) -> int:
    """MPEG-2 section CRC-32, byte-at-a-time against `CRC_TABLE` -- the
    exact algorithm go2rtc's own muxer uses for PAT/PMT section CRCs."""
    crc = 0xFFFFFFFF
    for b in data:
        crc = CRC_TABLE[(b ^ crc) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFF


class Bits:
    """Byte-aligned + bit writer matching go2rtc's `bits.Writer` usage here."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self._acc = 0
        self._nbits = 0

    def _flush_check(self) -> None:
        while self._nbits >= 8:
            self._nbits -= 8
            self.buf.append((self._acc >> self._nbits) & 0xFF)

    def bit(self, v: int, n: int) -> None:
        self._acc = (self._acc << n) | (v & ((1 << n) - 1))
        self._nbits += n
        self._flush_check()

    def byte(self, v: int) -> None:
        assert self._nbits == 0
        self.buf.append(v & 0xFF)

    def u16(self, v: int) -> None:
        assert self._nbits == 0
        self.buf.append((v >> 8) & 0xFF)
        self.buf.append(v & 0xFF)

    def bytes_(self, *vals: int) -> None:
        assert self._nbits == 0
        self.buf.extend(bytes(v & 0xFF for v in vals))

    def raw(self, b: bytes) -> None:
        assert self._nbits == 0
        self.buf.extend(b)

    def __len__(self) -> int:
        return len(self.buf)


def _psi_header(wr: Bits, table_id: int, size: int) -> None:
    wr.byte(0)          # pointer field
    wr.byte(table_id)   # table id
    wr.bit(1, 1)        # section syntax indicator
    wr.bit(0, 1)        # private bit
    wr.bit(0b11, 2)     # reserved
    wr.bit(0, 2)        # section length unused
    wr.bit(5 + size + 4, 10)  # section length
    wr.u16(1)           # table id extension
    wr.bit(0b11, 2)     # reserved
    wr.bit(0, 5)        # version
    wr.bit(1, 1)        # current/next
    wr.byte(0)          # section number
    wr.byte(0)          # last section number


def _ts_header(wr: Bits, pid: int) -> None:
    wr.byte(SYNC)
    wr.bit(0, 1)        # TEI
    wr.bit(1, 1)        # PUSI
    wr.bit(0, 1)        # priority
    wr.bit(pid, 13)
    wr.bit(0, 2)        # TSC
    wr.bit(0, 1)        # adaptation
    wr.bit(1, 1)        # payload
    wr.bit(0, 4)        # continuity counter


def _tail(wr: Bits) -> None:
    pad = PACKET_SIZE - (len(wr) % PACKET_SIZE)
    if pad != PACKET_SIZE:
        wr.raw(b"\x00" * pad)


def build_header() -> bytes:
    """The fixed PAT + PMT preamble every talk session sends once, before
    any audio: two 188-byte TS packets naming a single PCMATapo elementary
    stream on `PES0_PID`."""
    # PAT
    wr = Bits()
    _ts_header(wr, PAT_PID)
    i = len(wr) + 1
    _psi_header(wr, 0, 4)
    wr.u16(1)              # program num
    wr.bit(0b111, 3)
    wr.bit(PMT_PID, 13)
    crc = crc32_mpegts(bytes(wr.buf[i:]))
    wr.bytes_(crc & 0xFF, (crc >> 8) & 0xFF, (crc >> 16) & 0xFF, (crc >> 24) & 0xFF)
    _tail(wr)

    # PMT
    _ts_header(wr, PMT_PID)
    j = len(wr) + 1
    _psi_header(wr, 2, 4 + 1 * 5)
    wr.bit(0b111, 3)
    wr.bit(0x1FFF, 13)    # PCR PID (unused)
    wr.bit(0b1111, 4)
    wr.bit(0, 2)
    wr.bit(0, 10)         # program info length
    # one track
    wr.byte(STREAM_TYPE_PCMA_TAPO)
    wr.bit(0b111, 3)
    wr.bit(PES0_PID, 13)
    wr.bit(0b1111, 4)
    wr.bit(0, 2)
    wr.bit(0, 10)         # ES info length
    crc = crc32_mpegts(bytes(wr.buf[j:]))
    wr.bytes_(crc & 0xFF, (crc >> 8) & 0xFF, (crc >> 16) & 0xFF, (crc >> 24) & 0xFF)
    _tail(wr)
    return bytes(wr.buf)


def write_time(pts: int) -> bytes:
    """A PES optional-header PTS-only timestamp, 5 bytes, 90kHz clock."""
    b = bytearray(5)
    only_pts = 0x20
    b[0] = only_pts | ((pts >> (32 - 3)) & 0xFF) | 1
    b[1] = (pts >> (24 - 2)) & 0xFF
    b[2] = ((pts >> (16 - 2)) | 1) & 0xFF
    b[3] = (pts >> (8 - 1)) & 0xFF
    b[4] = ((pts << 1) | 1) & 0xFF
    return bytes(b)


class PesState:
    """Carries the running PTS/sequence counter across successive
    `get_payload()` calls for one talk session -- the muxer is
    stream-stateful across packets, never per-call."""

    def __init__(self) -> None:
        self.pts = 0
        self.timestamp = 0
        self.sequence = 0


def get_payload(pes: PesState, timestamp: int, payload: bytes) -> bytes:
    """Wrap `payload` (raw A-law bytes for one frame) in a PES header and
    split it across one or more 188-byte TS packets on `PES0_PID`."""
    if pes.timestamp != 0:
        pes.pts += (timestamp - pes.timestamp) & 0xFFFFFFFF
    pes.timestamp = timestamp

    size = 3 + 5 + len(payload)
    b = bytearray(6 + 3 + 5)
    b[0], b[1], b[2] = 0, 0, 1
    b[3] = STREAM_ID_AUDIO
    if size <= 0xFFFF:
        b[4] = (size >> 8) & 0xFF
        b[5] = size & 0xFF
    b[6] = 0x80
    b[7] = 0x80
    b[8] = 5
    b[9:14] = write_time(pes.pts)
    data = bytes(b) + payload

    out = bytearray()
    first = True
    while data:
        pid = PES0_PID
        hdr = bytearray()
        hdr.append(SYNC)
        pid_field = pid | (0x4000 if first else 0)
        hdr.append((pid_field >> 8) & 0xFF)
        hdr.append(pid_field & 0xFF)
        counter = pes.sequence & 0xF
        if len(data) < PACKET_SIZE - 4:
            hdr.append(0x30 | counter)  # adaptation + payload
            ad_size = PACKET_SIZE - 4 - 1 - len(data)
            hdr.append(ad_size)
            hdr.extend(b"\x00" * ad_size)
            hdr.extend(data)
            data = b""
        else:
            hdr.append(0x10 | counter)  # payload only
            hdr.extend(data[: PACKET_SIZE - 4])
            data = data[PACKET_SIZE - 4:]
        out.extend(hdr)
        pes.sequence += 1
        first = False
    return bytes(out)
