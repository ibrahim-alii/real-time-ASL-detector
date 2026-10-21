"""Single-request client for reusable workers in isolated Sol environments."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import suppress
from types import TracebackType
from typing import Any


class PersistentWorker:
    def __init__(self, command: list[str], *, timeout: float = 300,
                 env: dict[str, str] | None = None, stderr: Any = subprocess.DEVNULL) -> None:
        self.command, self.timeout = command, timeout
        self.lock = threading.Lock()
        self.messages: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, text=True, encoding="utf-8", bufsize=1, env=env,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            if self._receive(timeout) != {"ready": True, "protocol": 1}:
                raise RuntimeError("worker protocol handshake failed")
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                line = self.process.stdout.readline(65537)
                self.messages.put(line or None, timeout=1)
                if not line or len(line) > 65536:
                    break
        except (OSError, ValueError, queue.Full):
            return

    def _receive(self, timeout: float) -> dict[str, Any]:
        try:
            line = self.messages.get(timeout=timeout)
        except queue.Empty as error:
            self.close()
            raise subprocess.TimeoutExpired(self.command, timeout) from error
        try:
            if line is None or len(line) > 65536 or not line.endswith("\n"):
                raise ValueError("invalid line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("invalid object")
            return value
        except (ValueError, TypeError) as error:
            self.close()
            raise RuntimeError("worker protocol failed or process exited") from error

    def request(self, item: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("worker is busy")
        try:
            identifier = uuid.uuid4().hex
            line = json.dumps({**item, "id": identifier}, allow_nan=False) + "\n"
            if len(line) > 65536:
                raise ValueError("worker request exceeds protocol limit")
            if self.process.poll() is not None or self.process.stdin is None:
                raise RuntimeError("worker process is closed")
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                self.close()
                raise RuntimeError("worker pipe failed") from error
            reply = self._receive(self.timeout if timeout is None else timeout)
            if reply.get("id") != identifier or type(reply.get("ok")) is not bool:
                self.close()
                raise RuntimeError("worker response violates protocol")
            if not reply["ok"]:
                raise ValueError("worker rejected request: " + str(reply.get("error", "unknown")))
            if not isinstance(reply.get("result"), dict):
                self.close()
                raise RuntimeError("worker result violates protocol")
            return dict(reply["result"])
        finally:
            self.lock.release()

    def close(self) -> None:
        if self.process.poll() is None:
            if sys.platform == "win32":
                self.process.kill()
            else:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if threading.current_thread() is not self.reader:
            self.reader.join(timeout=2)
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self) -> PersistentWorker:
        return self

    def __exit__(self, _kind: type[BaseException] | None, _error: BaseException | None,
                 _trace: TracebackType | None) -> None:
        self.close()
