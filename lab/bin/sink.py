"""Attacker-side sink: accepts TCP on ports 20000-20999 and UDP on 30000-30999, discards data.
UDP queries get a 12-byte reply (like a DNS answer header)."""
import selectors, socket
sel = selectors.DefaultSelector()
for p in range(20000, 21000):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("10.10.2.2", p)); s.listen(64); s.setblocking(False); sel.register(s, selectors.EVENT_READ, "lst")
for p in range(30000, 31000):
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.bind(("10.10.2.2", p)); u.setblocking(False)
    sel.register(u, selectors.EVENT_READ, "udp")
while True:
    for key, _ in sel.select():
        s = key.fileobj
        if key.data == "lst":
            c, _ = s.accept(); c.setblocking(False); sel.register(c, selectors.EVENT_READ, "conn")
        elif key.data == "conn":
            try:
                d = s.recv(65536)
            except (ConnectionResetError, BlockingIOError):
                d = b""
            if not d:
                sel.unregister(s); s.close()
        else:
            try:
                d, a = s.recvfrom(65536); s.sendto(d[:12], a)
            except BlockingIOError:
                pass
