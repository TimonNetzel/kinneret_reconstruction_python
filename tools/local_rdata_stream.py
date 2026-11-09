"""Read XDR RData structure without materializing large numeric vectors.

Uses the installed rdata parser for object/reference semantics, but supplies a
gzip stream instead of its whole-file BytesIO. Skipped arrays are empty sentinels,
NOT usable scientific values. Only small children may be converted. No R executes.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import BinaryIO

import numpy as np
import rdata
from rdata.parser._parser import Parser, RObjectType
from rdata.parser._xdr import ParserXDR


class StreamXDR(ParserXDR):
    def __init__(self, stream: BinaryIO, max_vector_bytes: int = 131072) -> None:
        Parser.__init__(self, expand_altrep=False)
        self.file = stream
        self.max_vector_bytes = max_vector_bytes
        self.blocks: dict[int, dict] = {}
        self.retained_numeric_bytes = 0

    def _parse_array_values(self, dtype, length):
        dt = np.dtype(dtype)
        size = dt.itemsize * length
        if length < 0 or size > 2_000_000_000:
            raise ValueError("Unexpected numeric vector size")
        if size > self.max_vector_bytes:
            offset = self.file.tell()
            h = hashlib.sha256()
            remaining = size
            while remaining:
                block = self.file.read(min(remaining, 1048576))
                if not block:
                    raise EOFError("Truncated XDR numeric vector")
                h.update(block)
                remaining -= len(block)
            value = np.empty(0, dtype=dt)
            self.blocks[id(value)] = dict(length=length, byte_offset=offset,
                                          bytes=size, dtype=str(dt), xdr_sha256=h.hexdigest())
            return value
        if self.retained_numeric_bytes + size > 32_000_000:
            raise MemoryError("Small numeric arrays exceed this metadata reader's budget")
        raw = self.file.read(size)
        if len(raw) != size:
            raise EOFError("Truncated XDR numeric value")
        self.retained_numeric_bytes += size
        return np.frombuffer(raw, dtype=dt.newbyteorder(">")).astype(dt)


def read_structure(path: Path, max_vector_bytes: int = 131072):
    with path.open("rb") as f:
        magic = f.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as stream:
        header = stream.readline()
        if header not in [b"RDX2\n", b"RDX3\n"] or stream.readline() != b"X\n":
            raise ValueError("Only XDR RData v2/v3 is supported; no format guessing")
        parser = StreamXDR(stream, max_vector_bytes=max_vector_bytes)
        data = parser.parse_all()
        parser.check_complete()
    return data, parser


def resolve(obj):
    seen = set()
    while obj is not None and obj.info.type == RObjectType.REF:
        if id(obj) in seen:
            raise ValueError("Cyclic reference encountered while resolving a leaf")
        seen.add(id(obj))
        obj = obj.referenced_object
    return obj


def attributes(obj) -> dict:
    obj = resolve(obj)
    return rdata.conversion.convert(obj.attributes) if obj.attributes else {}


def children(obj) -> dict:
    obj = resolve(obj)
    if obj.info.type == RObjectType.LIST:
        result = {}
        while obj.info.type == RObjectType.LIST:
            tag = rdata.conversion.convert(obj.tag)
            result[str(tag)] = obj.value[0]
            obj = resolve(obj.value[1])
        return result
    if obj.info.type != RObjectType.VEC:
        raise ValueError("Expected an R list/vector")
    labels = attributes(obj).get("names", [str(i + 1) for i in range(len(obj.value))])
    if len(labels) != len(obj.value) or len(set(map(str, labels))) != len(labels):
        raise ValueError("Repeated or mismatched list keys require explicit handling")
    return dict(zip(map(str, labels), obj.value))


def small_value(obj, parser: StreamXDR):
    obj = resolve(obj)
    if id(obj.value) in parser.blocks:
        raise ValueError("Large numeric payload is deliberately absent; do not convert")
    return rdata.conversion.convert(obj)


def read_planes(path: Path, block: dict, dimensions: tuple[int, int, int]):
    if block["dtype"] != "float64" or int(np.prod(dimensions)) != block["length"]:
        raise ValueError("Dimensions do not describe the skipped double vector")
    nx, ny, nz = dimensions
    size = nx * ny * 8
    if size > 8_000_000:
        raise MemoryError("A plane exceeds the bounded reader's budget")
    with path.open("rb") as f:
        compressed = f.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    h = hashlib.sha256()
    with opener(path, "rb") as stream:
        stream.seek(block["byte_offset"])
        for i in range(nz):
            raw = stream.read(size)
            if len(raw) != size:
                raise EOFError("Truncated numeric plane")
            h.update(raw)
            yield i, np.frombuffer(raw, dtype=">f8").reshape((nx, ny), order="F")
    if h.hexdigest() != block["xdr_sha256"]:
        raise ValueError("Numeric payload hash changed between structure and plane reads")
