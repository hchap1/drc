#!/usr/bin/env python3
# monitor.py — run over SSH to watch jetson_collect.py
# Shows live log output and frame count for the current session.
#
# Usage:
#   python3 ~/drc/monitor.py

import os
import subprocess
import time
from pathlib import Path

LOG      = Path.home() / 'collect.log'
DATA_DIR = Path.home() / 'data'
TAIL     = 25   # log lines to show


def _frame_count(session_dir):
    frames = session_dir / 'frames'
    if not frames.exists():
        return 0
    return sum(1 for _ in frames.glob('*.jpg'))


def _is_running():
    try:
        out = subprocess.check_output(['pgrep', '-f', 'jetson_collect.py'])
        return out.strip().decode()
    except subprocess.CalledProcessError:
        return None


def main():
    while True:
        os.system('clear')

        pid = _is_running()
        status = f'RUNNING  pid={pid}' if pid else 'STOPPED'
        print(f'jetson_collect.py  [{status}]')
        print('─' * 50)

        # Latest session frame count
        sessions = sorted(DATA_DIR.glob('session_*'))
        if sessions:
            latest  = sessions[-1]
            n       = _frame_count(latest)
            print(f'Session : {latest.name}')
            print(f'Frames  : {n}')
        else:
            print('No sessions yet.')
        print('─' * 50)

        # Log tail
        if LOG.exists():
            lines = LOG.read_text(errors='replace').splitlines()
            for line in lines[-TAIL:]:
                print(line)
        else:
            print('(no log file — run:  nohup python3 ~/drc/jetson_collect.py > ~/collect.log 2>&1 &)')

        print()
        print('Ctrl-C to exit monitor  (does NOT stop jetson_collect.py)')

        time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
