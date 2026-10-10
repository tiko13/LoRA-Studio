"""Opt-in live integration check. Uses a neutral synthetic image, never starts training."""
import json
from pathlib import Path
import struct
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.integrations import PromptStudio
from studio.domain import normalize_project, caption_instruction, parse_caption

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

if __name__ == '__main__':
    out = Path('test-results/live-vision')
    out.mkdir(parents=True, exist_ok=True)
    data = fixture()
    (out/'sample.png').write_bytes(data)
    bridge = PromptStudio('http://127.0.0.1:8188')
    info = bridge.inspect()
    print(json.dumps({'available':info['available'], 'vision':info['vision']}, indent=2), flush=True)
    project = normalize_project({'name':'Neutral connection test', 'kind':'object', 'description':'The red rectangular shape on the left', 'trigger':'test_redshape'})
    started=time.time()
    result = parse_caption(bridge.caption(caption_instruction(project), data, 'sample.png', bridge.connection()), project)
    (out/'caption.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'seconds':round(time.time()-started,1), **result},indent=2),flush=True)
