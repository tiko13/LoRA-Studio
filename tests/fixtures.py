"""Synthetic, non-personal fixtures for the reproducible test suite."""
import struct
import zlib


def fixture():
    width, height = 512, 384
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            color = (238, 235, 225)
            if 70 <= x < 235 and 90 <= y < 275:
                color = (210, 46, 52)
            if (x-355)**2+(y-190)**2 < 76**2:
                color = (41, 95, 203)
            row.extend(color)
        rows.append(b'\0' + row)
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind+data))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')
