"""
data_server.py — run on LAPTOP to receive training data from the Jetson.
Creates a 'data/' folder in whatever directory you run this from.

Usage:
  python3 data_server.py

Then on the Jetson:
  python3 data_push.py <this-laptop-ip>
"""

import socket
import struct
import sys
from pathlib import Path

PORT  = 5010
CHUNK = 1 << 18   # 256 KB receive buffer
HDR   = struct.Struct('>HQ')   # path_len(uint16) + file_size(uint64)


def _local_ip():
    """Best-guess IP on the robot hotspot network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.168.4.1', 1))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def _recv_exact(sock, n):
    buf = bytearray(n)
    mv  = memoryview(buf)
    pos = 0
    while pos < n:
        got = sock.recv_into(mv[pos:], n - pos)
        if not got:
            raise ConnectionError('Jetson disconnected mid-transfer')
        pos += got
    return bytes(buf)


def main():
    out = Path('data')
    out.mkdir(exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('', PORT))
    srv.listen(1)
    print(f'Laptop IP : {_local_ip()}')
    print(f'Listening on port {PORT} — run  python3 data_push.py <laptop-ip>  on the Jetson')

    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    print(f'Jetson connected from {addr}')

    n_files, total_bytes = struct.unpack('>IQ', _recv_exact(conn, 12))
    print(f'{n_files} files  ({total_bytes / 1e6:.1f} MB)\n')

    received = 0
    for i in range(n_files):
        path_len, file_size = HDR.unpack(_recv_exact(conn, HDR.size))
        rel = _recv_exact(conn, path_len).decode()
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        remaining = file_size
        with open(dest, 'wb') as f:
            while remaining:
                chunk = _recv_exact(conn, min(CHUNK, remaining))
                f.write(chunk)
                remaining -= len(chunk)

        received += file_size
        pct = received / total_bytes * 100 if total_bytes else 100
        bar  = '#' * int(pct // 2)
        print(f'\r[{i+1:>{len(str(n_files))}}/{n_files}] {received/1e6:6.1f}/{total_bytes/1e6:.1f} MB  [{bar:<50}] {pct:5.1f}%',
              end='', flush=True)

    conn.close()
    srv.close()
    print(f'\n\nDone — {n_files} files saved to {out.resolve()}')


if __name__ == '__main__':
    main()
