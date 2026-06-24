"""
data_push.py — run on JETSON to push all training data to the laptop.
Sends everything under ../data/ (one level above the drc repo).

Usage:
  python3 data_push.py <laptop-ip>

The laptop must be running:
  python3 data_server.py
"""

import socket
import struct
import sys
from pathlib import Path

PORT     = 5010
CHUNK    = 1 << 18   # 256 KB read buffer (fallback if sendfile unavailable)
HDR      = struct.Struct('>HQ')   # path_len(uint16) + file_size(uint64)
DATA_DIR = Path(__file__).resolve().parent / '../data'


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 data_push.py <laptop-ip>')

    ip   = sys.argv[1]
    data = DATA_DIR.resolve()

    if not data.exists():
        sys.exit(f'Data directory not found: {data}')

    files = sorted(p for p in data.rglob('*') if p.is_file())
    if not files:
        sys.exit(f'No files found under {data}')

    total_bytes = sum(p.stat().st_size for p in files)
    print(f'Sending {len(files)} files  ({total_bytes / 1e6:.1f} MB)  →  {ip}:{PORT}')

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.connect((ip, PORT))
    print('Connected\n')

    # manifest header: file count (uint32) + total bytes (uint64)
    sock.sendall(struct.pack('>IQ', len(files), total_bytes))

    sent = 0
    for i, p in enumerate(files):
        rel       = str(p.relative_to(data)).encode()
        file_size = p.stat().st_size

        sock.sendall(HDR.pack(len(rel), file_size))
        sock.sendall(rel)

        with open(p, 'rb') as f:
            try:
                sock.sendfile(f)          # zero-copy kernel sendfile on Linux
            except AttributeError:
                while True:               # fallback for older Python
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    sock.sendall(chunk)

        sent += file_size
        pct  = sent / total_bytes * 100 if total_bytes else 100
        bar  = '#' * int(pct // 2)
        print(f'\r[{i+1:>{len(str(len(files)))}}/{len(files)}] {sent/1e6:6.1f}/{total_bytes/1e6:.1f} MB  [{bar:<50}] {pct:5.1f}%',
              end='', flush=True)

    sock.close()
    print(f'\n\nDone — {len(files)} files sent')


if __name__ == '__main__':
    main()
