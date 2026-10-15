"""Minimal TFRecord / tf.Event reader — dumps scalar curves without tensorboard."""
import glob
import struct
import sys


def read_varint(b, i):
    r = 0
    s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, i
        s += 7


def parse_value(data):
    i = 0
    tag = None
    sv = None
    while i < len(data):
        key, i = read_varint(data, i)
        ftype, wire = key >> 3, key & 7
        if wire == 0:
            _, i = read_varint(data, i)
        elif wire == 2:
            ln, i = read_varint(data, i)
            payload = data[i:i + ln]
            i += ln
            if ftype == 2:
                tag = payload.decode("utf8", "ignore")
        elif wire == 5:
            sv = struct.unpack_from("<f", data, i)[0]
            i += 4
        elif wire == 1:
            i += 8
        else:
            raise ValueError(wire)
    return tag, sv


def parse_summary(data):
    i = 0
    out = []
    while i < len(data):
        key, i = read_varint(data, i)
        ftype, wire = key >> 3, key & 7
        if wire == 0:
            _, i = read_varint(data, i)
        elif wire == 2:
            ln, i = read_varint(data, i)
            payload = data[i:i + ln]
            i += ln
            if ftype == 1:
                t, v = parse_value(payload)
                if t is not None and v is not None:
                    out.append((t, v))
        elif wire == 5:
            i += 4
        elif wire == 1:
            i += 8
        else:
            raise ValueError(wire)
    return out


def parse_event(data):
    i = 0
    step = None
    vals = []
    while i < len(data):
        key, i = read_varint(data, i)
        ftype, wire = key >> 3, key & 7
        if wire == 0:
            v, i = read_varint(data, i)
            if ftype == 2:
                step = v
        elif wire == 1:
            i += 8
        elif wire == 2:
            ln, i = read_varint(data, i)
            payload = data[i:i + ln]
            i += ln
            if ftype == 5:
                vals.extend(parse_summary(payload))
        elif wire == 5:
            i += 4
        else:
            raise ValueError(wire)
    return step, vals


def read_file(path):
    pts = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            ln = struct.unpack("<Q", hdr)[0]
            f.read(4)
            data = f.read(ln)
            if len(data) < ln:
                break
            f.read(4)
            try:
                step, vals = parse_event(data)
            except Exception:
                continue
            for tag, v in vals:
                pts.append((step or 0, tag, v))
    return pts


want = sys.argv[1] if len(sys.argv) > 1 else "val_loss"
files = sorted(glob.glob("/test21/conformer-rnnt-asr/exp_word_960h/lightning_logs/version_*/events.out.tfevents.*"))
print(f"[tb] {len(files)} event files")
pts = []
for fp in files:
    pts.extend(read_file(fp))
pts = [p for p in pts if want in p[1]]
pts.sort(key=lambda p: p[0])
seen = {}
for step, tag, v in pts:
    seen[step] = v
cur = sorted(seen.items())
print(f"[tb] {want}: {len(cur)} points")
for step, v in cur[-40:]:
    print(f"  step {step}: {v:.2f}")
