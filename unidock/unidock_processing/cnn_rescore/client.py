"""Client side of the resident CNN rescoring server (no torch / molgrid needed here).

The runner tries the server first (``UD2_CNN_SOCKET``, default ``/tmp/ud2_cnn_<uid>.sock``) and falls back to a fresh
process when no server answers. Protocol: one JSON object per line, request then reply, over a Unix socket.
"""
import getpass
import json
import os
import socket


def default_socket():
    return os.environ.get("UD2_CNN_SOCKET", os.path.join("/tmp", f"ud2_cnn_{getpass.getuser()}.sock"))


def rescore_via_server(socket_path, request, timeout=3600.0):
    """Send one rescoring request; returns the reply dict, or None when no server listens on socket_path."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(5.0)
        s.connect(socket_path)
    except OSError:
        s.close()
        return None
    try:
        s.settimeout(timeout)
        s.sendall((json.dumps(request) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                raise ConnectionError("CNN rescoring server closed the connection without a reply")
            buf += chunk
        return json.loads(buf.decode())
    finally:
        s.close()
