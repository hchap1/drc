# serial_client.py — REPLACED by motor_client.py
# USB serial to ESP32 has been removed in favour of WiFi UDP.
# All code should now do:  import motor_client
raise ImportError(
    'serial_client is removed — use motor_client instead.\n'
    '  import motor_client\n'
    '  motors = motor_client.connect()'
)
