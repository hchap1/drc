"""
run.py — control interface for cnn_backend.py.

Usage:
  python3 run.py start <folder> [speed_mult] [straight_mult] [--launch SECS]
  python3 run.py stop

`start` spawns cnn_backend.py if it isn't already running, then connects,
streams telemetry to stdout, and waits for Enter before arming the motors.
The backend (and motors) keep running after run.py exits.

`stop` connects to a running backend and disarms the motors immediately.
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import time

SOCK_PATH    = '/tmp/drc_cnn.sock'
CONNECT_WAIT = 60


def _connect(timeout):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            sock.connect(SOCK_PATH)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)
    sock.close()
    return None


def cmd_start(args):
    sock = _connect(timeout=1.0)

    if sock is None:
        backend = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_backend.py')
        subprocess.Popen(
            [sys.executable, backend, args.folder],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        print(f'[run] backend spawned ({args.folder}) — connecting...')
        sock = _connect(timeout=CONNECT_WAIT)
        if sock is None:
            print('[run] backend did not become available — did it crash on startup?')
            sys.exit(1)
    else:
        print('[run] connected to running backend')

    sock.sendall(
        f'START {args.speed_mult} {args.straight_mult} {args.launch}\n'.encode()
    )

    def _relay():
        try:
            for raw in sock.makefile('r'):
                line = raw.rstrip('\n')
                if line == 'READY':
                    print('[run] backend ready — press Enter to start driving...')
                elif line.startswith('LOG:'):
                    print(line[4:])
        except OSError:
            pass

    threading.Thread(target=_relay, daemon=True).start()

    input()

    try:
        sock.sendall(b'ARM\n')
    except OSError:
        print('[run] failed to send ARM — backend may have crashed')
        sys.exit(1)

    sock.close()
    print('[run] ARM sent — exiting (backend continues independently)')


def cmd_stop(args):
    sock = _connect(timeout=3.0)
    if sock is None:
        print(f'[run] no backend found at {SOCK_PATH}')
        sys.exit(1)
    try:
        sock.sendall(b'STOP\n')
    except OSError:
        print('[run] failed to send STOP')
        sys.exit(1)
    sock.close()
    print('[run] motors stopped')


def cmd_quit(args):
    sock = _connect(timeout=3.0)
    if sock is None:
        print(f'[run] no backend found at {SOCK_PATH}')
        sys.exit(1)
    try:
        sock.sendall(b'QUIT\n')
    except OSError:
        print('[run] failed to send QUIT')
        sys.exit(1)
    sock.close()
    print('[run] backend shutdown sent')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('start', help='arm the robot (spawning backend if needed)')
    p.add_argument('folder',        help='model folder containing model.pt')
    p.add_argument('speed_mult',    nargs='?', type=float, default=1.0,
                   help='motor output multiplier (default 1.0)')
    p.add_argument('straight_mult', nargs='?', type=float, default=1.0,
                   help='extra multiplier for straight sections (default 1.0)')
    p.add_argument('--launch', type=float, default=0.0, metavar='SECS',
                   help='blast at 0.7 for this many seconds before handing off to CNN')

    sub.add_parser('stop', help='disarm motors on a running backend')
    sub.add_parser('quit', help='shut down cnn_backend.py completely')

    args = ap.parse_args()

    if args.command == 'start':
        cmd_start(args)
    elif args.command == 'stop':
        cmd_stop(args)
    elif args.command == 'quit':
        cmd_quit(args)


if __name__ == '__main__':
    main()
