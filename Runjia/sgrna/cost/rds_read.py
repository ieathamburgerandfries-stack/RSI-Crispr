"""Minimal read-only parser for R's XDR serialisation (.rds), enough for plain data (lists, vectors, attributes). It executes nothing.

Used only to read `rules.rds` of the BoostMEC repository (github.com/oazarate/BoostMEC, commit cdf190c), which holds the integer code that each category of each
categorical feature had when the released model was trained: a feature such as the dinucleotide at positions 25-26 saw four categories in training and so is coded
1 to 4, not 1 to 16. Without these codes the released model cannot be run on new sequences (`cost_boostmec.py`).
"""
import gzip
import struct
import sys

NIL, REF = 254, 255
ALTREP = 238


class R:
    def __init__(self, b):
        self.b, self.p, self.refs = b, 0, []

    def i(self):
        v = struct.unpack(">i", self.b[self.p:self.p + 4])[0]; self.p += 4; return v

    def d(self):
        v = struct.unpack(">d", self.b[self.p:self.p + 8])[0]; self.p += 8; return v

    def item(self):
        flags = self.i()
        t, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
        if t == NIL or t == 242 or t == 241 or t == 253 or t == 252:       # NILVALUE, EMPTYENV, BASEENV, GLOBALENV, UNBOUNDVALUE
            return None
        if t == REF:
            k = flags >> 8
            if k == 0:
                k = self.i()
            return self.refs[k - 1]
        if t == 1:                                                            # SYMSXP
            s = self.item(); self.refs.append(s); return s
        if t == 2:                                                            # LISTSXP: a pairlist, read as a list of (tag, value)
            out = []
            while True:
                attr = self.item() if has_attr else None
                tag = self.item() if has_tag else None
                car = self.item()
                out.append((tag, car))
                flags = self.i()
                t2, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
                if t2 == NIL:
                    return out
                assert t2 == 2, f"unexpected cdr type {t2}"
        if t == 9:                                                            # CHARSXP
            n = self.i()
            if n == -1:
                return None
            s = self.b[self.p:self.p + n].decode("utf-8", "replace"); self.p += n; return s
        if t in (10, 13):                                                     # LGLSXP, INTSXP
            n = self.i(); v = [self.i() for _ in range(n)]
        elif t == 14:                                                         # REALSXP
            n = self.i(); v = [self.d() for _ in range(n)]
        elif t == 16:                                                         # STRSXP
            n = self.i(); v = [self.item() for _ in range(n)]
        elif t in (19, 20):                                                   # VECSXP, EXPRSXP
            n = self.i(); v = [self.item() for _ in range(n)]
        elif t == ALTREP:
            info = self.item(); state = self.item(); attr = self.item()
            cls = info[0][1] if isinstance(info, list) and info and isinstance(info[0], tuple) else info[0]
            cls = str(cls)
            if cls == "compact_intseq":
                n, start, step = state; v = [int(start + step * k) for k in range(int(n))]
            elif cls == "compact_realseq":
                n, start, step = state; v = [start + step * k for k in range(int(n))]
            elif cls == "deferred_string":
                arg = state[0] if isinstance(state, list) else state
                v = [str(int(x)) if float(x) == int(x) else str(x) for x in arg]
            else:
                raise ValueError("ALTREP class " + cls)
            return self._attach(v, attr)
        else:
            raise ValueError(f"type {t} at byte {self.p}")
        attr = self.item() if has_attr else None
        return self._attach(v, attr)

    @staticmethod
    def _attach(v, attr):
        names = None
        if isinstance(attr, list):
            for tag, val in attr:
                if tag == "names":
                    names = val
        return {"names": names, "values": v} if names is not None else v


def load(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    assert raw[:2] == b"X\n", raw[:4]
    r = R(raw); r.p = 2
    version = r.i(); r.i(); r.i()
    if version == 3:
        n = r.i(); r.p += n
    return r.item()


if __name__ == "__main__":
    obj = load(sys.argv[1])
    print(type(obj), (obj.keys() if isinstance(obj, dict) else len(obj)))
