"""Metadata-only extension for special R serialization objects absent in rdata.

R's serialize.c defines 252/250/248. Preserve distinct markers; never convert
or execute them. This does not restore environments, packages or predictions.
"""
from __future__ import annotations
import enum
import gzip
import struct
from collections import Counter, OrderedDict
from pathlib import Path
from rdata.parser._parser import RObject, RObjectInfo, RObjectType, BYTECODE_SPECIAL_SET
from local_rdata_stream import StreamXDR


class MetadataSpecial(enum.Enum):
    UNBOUNDVALUE = 252
    BASENAMESPACE = 250
    PACKAGE = 248


_BYTECODE_TAGS = frozenset(x.value for x in BYTECODE_SPECIAL_SET)


class MetadataXDR(StreamXDR):
    def __init__(self, stream, max_vector_bytes=131072, *,
                 max_objects=2_000_000, max_parsed_objects=10_000_000,
                 max_cached_characters=65536):
        super().__init__(stream, max_vector_bytes)
        self.pending_int = None
        self.special_counts = Counter()
        self.object_count = 0
        self.max_objects = max_objects
        self.max_parsed_objects = max_parsed_objects
        self.max_cached_characters = max_cached_characters
        # Real caret containers repeat 47,323-entry character row-name
        # vectors. The bounded cache covers those leaves without increasing
        # the retained-object or numeric-payload budget.
        self.character_cache = OrderedDict()
        self.reused_characters = 0

    def parse_int(self):
        if self.pending_int is not None:
            value, self.pending_int = self.pending_int, None
            return value
        raw = self.file.read(4)
        if len(raw) != 4:
            raise EOFError("Truncated XDR integer")
        return struct.unpack(">i", raw)[0]

    def parse_R_object(self, reference_list=None, bytecode_rep_list=None, info_int=None):
        self.object_count += 1
        if self.object_count > self.max_parsed_objects:
            raise MemoryError('Metadata parsed-object work budget exceeded')
        if self.object_count - self.reused_characters > self.max_objects:
            raise MemoryError('Metadata retained-object budget exceeded')
        if info_int is not None:
            # R ReadBCConsts/ReadBCLang: only bytecode-special tags consume
            # their payload here. All other tags are followed by ReadItem.
            if info_int not in _BYTECODE_TAGS:
                return self.parse_R_object(reference_list)
            return super().parse_R_object(reference_list, bytecode_rep_list, info_int)
        # ReadItem starts a new bytecode repetition scope. Only the compact
        # ReadBCConsts/ReadBCLang branches above share the enclosing table.
        # Passing that table into an ordinary nested BCODESXP skips its own
        # repetition-count word and desynchronizes the rest of the stream.
        bytecode_rep_list = None
        raw = self.parse_int()
        if raw in (252, 250, 248):
            kind = MetadataSpecial(raw)
            self.special_counts[kind.name] += 1
            info = RObjectInfo(type=kind, object=False, attributes=False, tag=False, gp=0, reference=0)
            if kind == MetadataSpecial.PACKAGE:
                if self.parse_int() != 0:
                    raise ValueError('Unsupported named package reference')
                value = self._parse_vector_value(reference_list, bytecode_rep_list)
            else:
                value = kind.name
            obj = RObject(info=info, tag=None, attributes=None, value=value, referenced_object=None)
            if kind == MetadataSpecial.PACKAGE and reference_list is not None:
                reference_list.append(obj)
            return obj
        if info_int is None:
            self.pending_int = raw
        obj = super().parse_R_object(reference_list, bytecode_rep_list, info_int)
        # CHAR objects are immutable leaves. R writes repeated strings in full;
        # interning bounded, short, attribute-free leaves preserves every value
        # and flag while preventing millions of duplicate metadata allocations.
        if (obj.info.type == RObjectType.CHAR and obj.attributes is None
                and obj.tag is None and isinstance(obj.value, bytes)
                and len(obj.value) <= 128):
            key = (raw, obj.value)
            previous = self.character_cache.get(key)
            if previous is not None:
                self.reused_characters += 1
                self.character_cache.move_to_end(key)
                return previous
            if self.max_cached_characters > 0:
                if len(self.character_cache) >= self.max_cached_characters:
                    self.character_cache.popitem(last=False)
                self.character_cache[key] = obj
        return obj


def read_metadata(path: Path, max_vector_bytes=131072):
    with path.open('rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as stream:
        if stream.readline() not in (b'RDX2\n', b'RDX3\n') or stream.readline() != b'X\n':
            raise ValueError('Only XDR RData v2/v3 supported')
        parser = MetadataXDR(stream, max_vector_bytes)
        data = parser.parse_all()
        parser.check_complete()
    return data, parser
