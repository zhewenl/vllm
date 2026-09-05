# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acknowledged control messages for engines without standalone notifications."""

import queue
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future

import zmq


class ControlChannel:
    """Own all sockets on one I/O thread, including sends from transfer threads."""

    def __init__(self, hostname: str, timeout: float = 5.0):
        self.name = uuid.uuid4().hex
        self.timeout = timeout
        self._outgoing: queue.Queue[tuple[str, bytes, Future[None]]] = queue.Queue()
        self._incoming: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self._disconnecting: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._ready: Future[str] = Future()
        self._thread = threading.Thread(
            target=self._run,
            args=(hostname,),
            daemon=True,
            name="p2p-control",
        )
        self._thread.start()
        self.address = self._ready.result(timeout=timeout)

    def send(self, address: str, payload: bytes) -> None:
        if self._stop.is_set():
            raise RuntimeError("P/D control channel is closed")
        future: Future[None] = Future()
        self._outgoing.put((address, payload, future))
        future.result(timeout=self.timeout + 1)

    def receive(self) -> dict[str, list[bytes]]:
        result: dict[str, list[bytes]] = defaultdict(list)
        while True:
            try:
                sender, payload = self._incoming.get_nowait()
            except queue.Empty:
                return dict(result)
            result[sender].append(payload)

    def disconnect(self, address: str) -> None:
        self._disconnecting.put(address)

    def close(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self, hostname: str) -> None:
        context = zmq.Context()
        router = context.socket(zmq.ROUTER)
        sockets: dict[str, zmq.Socket] = {}
        pending: dict[bytes, tuple[Future[None], float]] = {}
        poller = zmq.Poller()
        try:
            host = f"[{hostname}]" if ":" in hostname else hostname
            if ":" in hostname:
                router.setsockopt(zmq.IPV6, 1)
            port = router.bind_to_random_port(f"tcp://{host}")
            poller.register(router, zmq.POLLIN)
            self._ready.set_result(f"tcp://{host}:{port}")
            while not self._stop.is_set():
                while True:
                    try:
                        address = self._disconnecting.get_nowait()
                    except queue.Empty:
                        break
                    socket = sockets.pop(address, None)
                    if socket is not None:
                        poller.unregister(socket)
                        socket.close(linger=0)
                # Bound work so a busy producer cannot starve inbound messages.
                for _ in range(64):
                    try:
                        address, payload, future = self._outgoing.get_nowait()
                    except queue.Empty:
                        break
                    socket = sockets.get(address)
                    if socket is None:
                        socket = context.socket(zmq.DEALER)
                        socket.setsockopt(zmq.IPV6, 1)
                        socket.connect(address)
                        sockets[address] = socket
                        poller.register(socket, zmq.POLLIN)
                    message_id = uuid.uuid4().bytes
                    try:
                        socket.send_multipart(
                            [message_id, self.name.encode(), payload], flags=zmq.NOBLOCK
                        )
                    except zmq.ZMQError as exc:
                        future.set_exception(exc)
                    else:
                        pending[message_id] = (future, time.monotonic() + self.timeout)

                for socket in dict(poller.poll(5)):
                    frames = socket.recv_multipart()
                    if socket is router:
                        if len(frames) != 4:
                            continue
                        identity, message_id, sender, payload = frames
                        self._incoming.put((sender.decode(), payload))
                        router.send_multipart([identity, message_id])
                    elif len(frames) == 1:
                        entry = pending.pop(frames[0], None)
                        if entry is not None:
                            entry[0].set_result(None)
                now = time.monotonic()
                for message_id, (future, deadline) in list(pending.items()):
                    if deadline < now:
                        pending.pop(message_id)
                        future.set_exception(TimeoutError("P/D notification timed out"))
        except Exception as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            for future, _ in pending.values():
                future.set_exception(exc)
            pending.clear()
        finally:
            self._stop.set()
            for future, _ in pending.values():
                future.set_exception(RuntimeError("P/D control channel closed"))
            while not self._outgoing.empty():
                _, _, future = self._outgoing.get_nowait()
                future.set_exception(RuntimeError("P/D control channel closed"))
            router.close(linger=0)
            for socket in sockets.values():
                socket.close(linger=0)
            context.term()
