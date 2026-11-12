"""Minimal GGUF v3 reader/writer (no dependencies beyond numpy): key/values kept as raw bytes unless replaced,
tensors as (name, dims, ggml type, bytes)."""
import struct

import numpy as np

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
TYPE_SIZE = {0: 4, 1: 2}  # F32, F16 (bytes per element); other types are copied as bytes


class GGUF:
    def __init__(self, path):
        data = open(path, "rb").read()
        self.pos = 0
        self.data = data
        magic, version = self._take("I"), self._take("I")
        assert magic == 0x46554747 and version == 3, (hex(magic), version)
        nt, nkv = self._take("Q"), self._take("Q")
        self.kv = {}  # key -> (type, raw value bytes)
        self.alignment = 32
        for _ in range(nkv):
            k = self._string(); t = self._take("I"); v0 = self.pos
            self._skip(t)
            self.kv[k] = (t, data[v0:self.pos])
            if k == "general.alignment":
                self.alignment = struct.unpack_from("<I", data, v0)[0]
        infos = []
        for _ in range(nt):
            name = self._string(); nd = self._take("I"); dims = [self._take("Q") for _ in range(nd)]
            t = self._take("I"); off = self._take("Q")
            infos.append((name, dims, t, off))
        start = (self.pos + self.alignment - 1) // self.alignment * self.alignment
        offs = sorted(o for *_, o in infos) + [len(data) - start]
        self.tensors = {}  # name -> (dims, type, bytes), in file order
        for name, dims, t, off in infos:
            n = int(np.prod(dims)) if dims else 1
            size = n * TYPE_SIZE[t] if t in TYPE_SIZE else offs[offs.index(off) + 1] - off
            self.tensors[name] = (dims, t, data[start + off:start + off + size])
        del self.data

    def _take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.data, self.pos); self.pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def _string(self):
        n = self._take("Q"); s = self.data[self.pos:self.pos + n].decode(); self.pos += n
        return s

    def _skip(self, t):
        if t == 8:
            self._string(); return
        if t == 9:
            et, n = self._take("I"), self._take("Q")
            if et in SC:
                self.pos += n * struct.calcsize("<" + SC[et]); return
            for _ in range(n):
                self._skip(et)
            return
        self._take(SC[t])

    # values
    def get_str(self, k, default=None):
        if k not in self.kv: return default
        t, raw = self.kv[k]; assert t == 8
        n = struct.unpack_from("<Q", raw)[0]
        return raw[8:8 + n].decode()

    def get_int(self, k, default=None):
        if k not in self.kv: return default
        t, raw = self.kv[k]
        return struct.unpack_from("<" + SC[t], raw)[0]

    def get_array(self, k):
        t, raw = self.kv[k]; assert t == 9
        et, n = struct.unpack_from("<IQ", raw)
        return list(struct.unpack_from("<" + SC[et] * n, raw, 12))

    def set_str(self, k, s):
        b = s.encode(); self.kv[k] = (8, struct.pack("<Q", len(b)) + b)

    def set_i32(self, k, v):
        self.kv[k] = (5, struct.pack("<i", v))

    def set_array(self, k, values, et):  # et: 5 int32, 6 float32
        self.kv[k] = (9, struct.pack("<IQ", et, len(values)) + struct.pack("<" + SC[et] * len(values), *values))

    def array(self, name):
        dims, t, b = self.tensors[name]; assert t == 0, name
        return np.frombuffer(b, dtype=np.float32).reshape(list(reversed(dims)))

    def set_tensor(self, name, arr, after=None):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        item = (list(reversed(arr.shape)) if arr.ndim else [1], 0, arr.tobytes())
        if after is None or name in self.tensors:
            self.tensors[name] = item; return
        items = list(self.tensors.items()); i = [k for k, _ in items].index(after)
        items.insert(i + 1, (name, item)); self.tensors = dict(items)

    def write(self, path):
        out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(self.tensors), len(self.kv)))
        for k, (t, raw) in self.kv.items():
            kb = k.encode(); out += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t) + raw
        off, blobs = 0, []
        for name, (dims, t, b) in self.tensors.items():
            nb = name.encode()
            out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            out += struct.pack("<IQ", t, off)
            blobs.append((off, b)); off += (len(b) + self.alignment - 1) // self.alignment * self.alignment
        pad = (-len(out)) % self.alignment
        out += b"\0" * pad
        body = bytearray(off)
        for o, b in blobs:
            body[o:o + len(b)] = b
        with open(path, "wb") as f:
            f.write(out); f.write(body)
