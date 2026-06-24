"""
jetson_collect.py — controller-driven data collection on the Jetson.

USB Xbox-style controller plugged into the Jetson drives the robot and
toggles recording. No laptop connection required once started.

Controls:
  Right trigger        throttle  (0 → 0.30 forward power)
  Right stick X        steering
  Left trigger > 50%   recording active (hold to capture)

  Controller vibrates when recording starts (strong) and stops (soft).

Launch via SSH — keeps running after the connection drops:
  nohup python3 jetson_collect.py > ~/collect.log 2>&1 &

Stop:
  kill $(pgrep -f jetson_collect.py)

Copy data to laptop after the session:
  scp -r user@192.168.4.1:~/drc/data ./data

Prerequisites on Jetson:
  pip3 install evdev
  sudo usermod -a -G input $USER   # then log out and back in
"""

import csv
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import evdev
from evdev import ecodes, ff

import serial_client as motor_client
from cnn_model import IMG_W, IMG_H

# ── Motor mixing ───────────────────────────────────────────────────────────────
THROTTLE_MAX        = 0.30   # max forward power at full trigger
STEER_MAX           = 0.30   # max per-wheel offset at full stick deflection
STEER_INNER_PENALTY = 0.50   # extra reduction on the inside wheel while turning

# ── Camera ─────────────────────────────────────────────────────────────────────
JPEG_Q      = 90
SENSOR_W    = 1280
SENSOR_H    = 720
FRAMERATE   = 30
FLIP_METHOD = 2

# ── Controller axis codes (xpad / HID-Xbox driver on Linux) ───────────────────
_ABS_LEFT_TRIGGER  = ecodes.ABS_Z
_ABS_RIGHT_TRIGGER = ecodes.ABS_RZ
_ABS_RIGHT_STICK_X = ecodes.ABS_RX
_STICK_DEAD_ZONE   = 0.08    # ignore deflection below this fraction of full range

# ── Shared state (written by controller thread, read by main loop) ─────────────
_ctrl      = {'throttle': 0.0, 'steering': 0.0, 'recording': False}
_ctrl_lock = threading.Lock()

# ── Shared camera state (written by capture thread, read by main loop) ─────────
_latest_frame = None
_frame_id     = 0
_frame_lock   = threading.Lock()

_running = True


# ── GStreamer pipeline ──────────────────────────────────────────────────────────

def _pipeline():
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width={SENSOR_W}, height={SENSOR_H}, framerate={FRAMERATE}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw(memory:NVMM), width={IMG_W}, height={IMG_H} ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


# ── Camera capture thread ───────────────────────────────────────────────────────

def _capture_loop(cap):
    global _latest_frame, _frame_id
    while _running:
        ok, frame = cap.read()
        if ok:
            with _frame_lock:
                _latest_frame = frame
                _frame_id    += 1


# ── Controller ──────────────────────────────────────────────────────────────────

def _find_controller():
    """Return the first evdev device that has ABS triggers and force-feedback."""
    for path in evdev.list_devices():
        try:
            dev  = evdev.InputDevice(path)
            caps = dev.capabilities()
            abs_codes = caps.get(ecodes.EV_ABS, [])
            has_triggers = (
                _ABS_LEFT_TRIGGER  in abs_codes and
                _ABS_RIGHT_TRIGGER in abs_codes
            )
            has_ff = ecodes.EV_FF in caps
            if has_triggers and has_ff:
                return dev
            dev.close()
        except Exception:
            pass
    return None


def _norm_trigger(value, absinfo):
    """Raw trigger value → [0.0, 1.0]."""
    span = absinfo.max - absinfo.min
    return (value - absinfo.min) / span if span else 0.0


def _norm_axis(value, absinfo):
    """Raw stick value → [-1.0, 1.0]."""
    lo, hi = absinfo.min, absinfo.max
    half   = (hi - lo) / 2.0
    mid    = lo + half
    return (value - mid) / half if half else 0.0


def _rumble(device, kind):
    """
    Play a brief vibration effect.
    kind='start' → strong long pulse; kind='stop' → weak short pulse.
    Runs in its own thread so it never blocks the event loop.
    """
    try:
        strong, weak, ms = (0xFFFF, 0x0000, 300) if kind == 'start' else (0x0000, 0xFFFF, 150)
        effect = ff.Effect(
            ff.FF_RUMBLE, -1, 0,
            ff.Trigger(0, 0),
            ff.Replay(ms, 0),
            ff.EffectType(ff_rumble_effect=ff.Rumble(
                strong_magnitude=strong, weak_magnitude=weak
            )),
        )
        eid = device.upload_effect(effect)
        device.write(ecodes.EV_FF, eid, 1)
    except Exception as e:
        print(f'[rumble] {e}')


def _controller_loop(device):
    """
    Read evdev events and update _ctrl state.
    Detects left-trigger threshold crossings and fires rumble + log messages.
    """
    global _running
    caps_abs  = dict(device.capabilities(absval=True).get(ecodes.EV_ABS, []))
    was_rec   = False

    try:
        for event in device.read_loop():
            if not _running:
                break
            if event.type != ecodes.EV_ABS:
                continue

            code = event.code
            info = caps_abs.get(code)
            if info is None:
                continue

            if code == _ABS_RIGHT_TRIGGER:
                with _ctrl_lock:
                    _ctrl['throttle'] = _norm_trigger(event.value, info)

            elif code == _ABS_LEFT_TRIGGER:
                recording = _norm_trigger(event.value, info) > 0.5
                with _ctrl_lock:
                    _ctrl['recording'] = recording
                if recording != was_rec:
                    threading.Thread(
                        target=_rumble,
                        args=(device, 'start' if recording else 'stop'),
                        daemon=True,
                    ).start()
                    print(f'[rec] {"ON" if recording else "OFF"}')
                    was_rec = recording

            elif code == _ABS_RIGHT_STICK_X:
                val = _norm_axis(event.value, info)
                if abs(val) < _STICK_DEAD_ZONE:
                    val = 0.0
                with _ctrl_lock:
                    _ctrl['steering'] = val

    except OSError as e:
        print(f'[controller] disconnected: {e}')
        _running = False


# ── Motor mixing ────────────────────────────────────────────────────────────────

def _compute_motors(throttle, steering):
    """
    throttle: [0, 1]   → scales to THROTTLE_MAX
    steering: [-1, 1]  → outside wheel boosted, inside wheel reduced + penalised

    At full right stick (steering=+1, throttle=1):
      left  = 0.30 + 0.30         = 0.60
      right = 0.30 - 0.30 - 0.50 = -0.50  (inside wheel reverses for tight turn)
    """
    base   = throttle * THROTTLE_MAX
    offset = steering * STEER_MAX
    inner  = abs(steering) * STEER_INNER_PENALTY

    if steering >= 0:   # turning right: right wheel is inside
        left  = base + offset
        right = base - offset - inner
    else:               # turning left: left wheel is inside
        left  = base + offset - inner
        right = base - offset

    return max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right))


# ── Frame save thread ───────────────────────────────────────────────────────────

def _flush(csv_path, rows):
    if not rows:
        return
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['frame', 'left', 'right'])
        w.writerows(rows)
    rows.clear()


def _save_loop(save_q, fdir, csv_path):
    """
    Dequeue (fname, frame, left, right) tuples and write to disk.
    Receives None as a sentinel to flush remaining rows and exit cleanly.
    Runs as a daemon so disk I/O never stalls the control loop.
    """
    rows = []
    while True:
        item = save_q.get()
        if item is None:
            _flush(csv_path, rows)
            save_q.task_done()
            break
        fname, frame, left, right = item
        ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        if ok:
            (fdir / fname).write_bytes(enc.tobytes())
            rows.append((fname, round(left, 4), round(right, 4)))
        if len(rows) >= 200:
            _flush(csv_path, rows)
        save_q.task_done()


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    global _running

    # ── Controller ──────────────────────────────────────────────────────────────
    controller = _find_controller()
    if controller is None:
        sys.exit(
            'No gamepad found.\n'
            '  • Check the USB connection.\n'
            '  • Ensure the xpad driver is loaded: sudo modprobe xpad\n'
            '  • Ensure your user is in the input group: sudo usermod -a -G input $USER'
        )
    print(f'[ctrl] {controller.name}  ({controller.path})')

    # ── Camera ───────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera — check ribbon cable and pipeline settings')

    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    print('Waiting for first camera frame...')
    while True:
        with _frame_lock:
            if _latest_frame is not None:
                break
        time.sleep(0.05)
    print(f'[cam] ready  ({IMG_W}×{IMG_H})')

    # ── Motors ───────────────────────────────────────────────────────────────────
    motors = motor_client.connect()
    print('[motor] ESP32 connected via serial')

    # ── Session directory ─────────────────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    sess     = Path(__file__).resolve().parent / f'../data/session_{ts}'
    fdir     = sess / 'frames'
    csv_path = sess / 'labels.csv'
    fdir.mkdir(parents=True, exist_ok=True)
    print(f'[sess] saving to {sess}')

    # ── Save thread ───────────────────────────────────────────────────────────────
    save_q = queue.Queue(maxsize=120)   # ~4 s buffer at 30 fps; drops frames if full
    threading.Thread(target=_save_loop, args=(save_q, fdir, csv_path), daemon=True).start()

    # ── Controller thread ─────────────────────────────────────────────────────────
    threading.Thread(target=_controller_loop, args=(controller,), daemon=True).start()

    # ── Shutdown handler ──────────────────────────────────────────────────────────
    def _shutdown(sig, _frame):
        global _running
        print('\n[shutdown] stopping...')
        _running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Control loop ──────────────────────────────────────────────────────────────
    frame_idx     = 0
    last_saved_id = -1
    was_recording = False
    t0            = time.monotonic()
    t_log         = t0

    print('Running. Hold left trigger to record. SIGTERM or Ctrl-C to stop.')

    try:
        while _running:
            with _ctrl_lock:
                throttle  = _ctrl['throttle']
                steering  = _ctrl['steering']
                recording = _ctrl['recording']

            left, right = _compute_motors(throttle, steering)
            motors.send(left, right)

            if recording:
                with _frame_lock:
                    frame = _latest_frame
                    fid   = _frame_id

                if frame is not None and fid != last_saved_id:
                    fname = f'{frame_idx:06d}.jpg'
                    try:
                        save_q.put_nowait((fname, frame.copy(), left, right))
                        last_saved_id = fid
                        frame_idx    += 1
                    except queue.Full:
                        pass   # save thread is behind, drop this frame rather than block

            if was_recording and not recording:
                print(f'[rec] {frame_idx} frames saved so far')
            was_recording = recording

            now = time.monotonic()
            if now - t_log >= 5.0:
                print(
                    f'thr={throttle:.2f}  steer={steering:+.2f}  '
                    f'L={left:+.3f}  R={right:+.3f}  '
                    f'rec={recording}  frames={frame_idx}'
                )
                t_log = now

            time.sleep(0.01)   # 100 Hz control loop

    finally:
        _running = False
        motors.send(0.0, 0.0)
        motors.close()
        save_q.put(None)   # sentinel: flush remaining rows and exit save thread
        save_q.join()      # wait until all queued frames are written
        cap.release()
        print(f'[done] {frame_idx} frames → {sess}')
        print(f'Copy:   scp -r user@192.168.4.1:~/drc/data ./data')


if __name__ == '__main__':
    main()
