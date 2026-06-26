"""
run.py — control interface for cnn_backend.py.

Usage:
  python3 run.py launch <folder>
  python3 run.py start  <speed_mult> <straight_mult> <launch_secs>
  python3 run.py stop
  python3 run.py quit
"""

import argparse
import os
import socket
import subprocess
import sys
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


def _require_backend():
    sock = _connect(timeout=2.0)
    if sock is None:
        print('[run] no backend running — use: python3 run.py launch <folder>')
        sys.exit(1)
    return sock


def cmd_launch(args):
    sock = _connect(timeout=1.0)

    if sock is None:
        backend  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_backend.py')
        log_path = '/tmp/drc_backend.log'
        log_file = open(log_path, 'w')
        subprocess.Popen(
            [sys.executable, backend, args.folder],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        log_file.close()
        print('[run] backend spawned — log: {}'.format(log_path))
        print('[run] waiting for init...')
        sock = _connect(timeout=CONNECT_WAIT)
        if sock is None:
            print('[run] backend did not start in time — did it crash?')
            sys.exit(1)
    else:
        print('[run] backend already running')

    try:
        for raw in sock.makefile('r'):
            line = raw.rstrip('\n')
            if line == 'READY':
                print('[run] ready — use: python3 run.py start <speed> <straight> <launch>')
                break
            elif line.startswith('LOG:'):
                print(line[4:])
    except OSError:
        pass

    sock.close()


def cmd_start(args):
    sock = _require_backend()
    try:
        msg = 'START {s} {st} {l}\nARM\n'.format(
            s=args.speed_mult, st=args.straight_mult, l=args.launch)
        sock.sendall(msg.encode())
    except OSError:
        print('[run] failed to send — backend may have crashed')
        sys.exit(1)
    sock.close()
    print('[run] started')


def cmd_stop(args):
    sock = _require_backend()
    try:
        sock.sendall(b'STOP\n')
    except OSError:
        print('[run] failed to send STOP')
        sys.exit(1)
    sock.close()
    print('[run] motors stopped')


def cmd_quit(args):
    sock = _require_backend()
    try:
        sock.sendall(b'QUIT\n')
    except OSError:
        print('[run] failed to send QUIT')
        sys.exit(1)
    sock.close()
    print('[run] backend shut down')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command')

    p = sub.add_parser('launch', help='start the backend and wait for it to be ready')
    p.add_argument('folder', help='model folder containing model.pt')

    p = sub.add_parser('start', help='arm motors with given parameters (backend must be launched first)')
    p.add_argument('speed_mult',    nargs='?', type=float, default=1.0,
                   help='motor output multiplier (default 1.0)')
    p.add_argument('straight_mult', nargs='?', type=float, default=1.0,
                   help='extra multiplier for straight sections (default 1.0)')
    p.add_argument('launch',        nargs='?', type=float, default=0.0,
                   help='seconds to blast at 0.7 before CNN takes over (default 0.0)')

    sub.add_parser('stop', help='disarm motors, keep backend running')
    sub.add_parser('quit', help='shut down the backend completely')

    args = ap.parse_args()

    if not args.command:
        ap.print_help()
        sys.exit(1)

    if args.command == 'launch':
        cmd_launch(args)
    elif args.command == 'start':
        cmd_start(args)
    elif args.command == 'stop':
        cmd_stop(args)
    elif args.command == 'quit':
        cmd_quit(args)


if __name__ == '__main__':
    main()
