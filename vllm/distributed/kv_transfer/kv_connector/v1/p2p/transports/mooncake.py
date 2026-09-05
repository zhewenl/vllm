# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Mooncake READ/WRITE transport, independent of the NIXL package."""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from ..transport import (
    FatalTransferError,
    Operation,
    Regions,
    TransferState,
    TransferTelemetry,
    TransferTransport,
)
from .control import ControlChannel

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class MooncakePeer:
    version: int
    name: str
    session: str
    control_address: str


@dataclass(frozen=True)
class PreparedDescriptors:
    peer: str | None
    regions: np.ndarray


@dataclass(eq=False)
class MooncakeTransfer:
    operation: Operation
    peer: MooncakePeer
    local_addresses: list[int]
    remote_addresses: list[int]
    lengths: list[int]
    notification: bytes
    future: Future[TransferTelemetry] | None = None


class MooncakeTransport(TransferTransport):
    def __init__(self, config: "VllmConfig"):
        import torch
        from mooncake.engine import TransferEngine

        from vllm.platforms import current_platform
        from vllm.utils.network_utils import get_ip

        assert (transfer_config := config.kv_transfer_config) is not None
        self._device_id = (
            torch.accelerator.current_device_index()
            if current_platform.device_type != "cpu"
            else None
        )
        if transfer_config.kv_buffer_device not in ("cpu", "cuda"):
            raise ValueError("Mooncake P/D supports CPU and CUDA transfer buffers")
        hostname = transfer_config.get_from_extra_config("mooncake_hostname", None)
        hostname = hostname or get_ip()
        self.engine = TransferEngine()
        try:
            ret = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                transfer_config.get_from_extra_config("mooncake_protocol", "rdma"),
                transfer_config.get_from_extra_config("device_name", ""),
            )
        finally:
            # Mooncake's NVLink initialization can change the current device.
            if self._device_id is not None:
                current_platform.set_device(self._device_id)
        if ret != 0:
            raise RuntimeError(f"Mooncake initialization failed: {ret}")
        for method in ("batch_transfer_sync_read", "batch_transfer_sync_write"):
            if not callable(getattr(self.engine, method, None)):
                raise RuntimeError(f"Installed Mooncake does not support {method}")
        self._control = ControlChannel(
            hostname,
            timeout=transfer_config.get_from_extra_config("notification_timeout", 5.0),
        )
        self._peer = MooncakePeer(
            1,
            self._control.name,
            f"{hostname}:{self.engine.get_rpc_port()}",
            self._control.address,
        )
        self._peers: dict[str, MooncakePeer] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=transfer_config.get_from_extra_config("num_workers", 10),
            initializer=self._bind_device,
            thread_name_prefix="mooncake-transfer",
        )
        self._active: set[MooncakeTransfer] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._fatal_error: FatalTransferError | None = None

    def _bind_device(self) -> None:
        if self._device_id is not None:
            from vllm.platforms import current_platform

            current_platform.set_device(self._device_id)

    def register_memory(self, regions: Regions) -> tuple[list[int], list[int]]:
        addresses = [int(region[0]) for region in regions]
        lengths = [int(region[1]) for region in regions]
        ret = self.engine.batch_register_memory(addresses, lengths)
        if ret != 0:
            raise RuntimeError(f"Mooncake memory registration failed: {ret}")
        return addresses, lengths

    def unregister_memory(self, registration: tuple[list[int], list[int]]) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error
        ret = self.engine.batch_unregister_memory(registration[0])
        if ret != 0:
            raise RuntimeError(f"Mooncake memory unregistration failed: {ret}")

    def export_peer(self) -> bytes:
        return msgspec.msgpack.encode(self._peer)

    def connect_peer(self, metadata: bytes) -> str:
        peer = msgspec.msgpack.decode(metadata, type=MooncakePeer)
        if peer.version != 1:
            raise ValueError(f"Unsupported Mooncake transport version: {peer.version}")
        with self._lock:
            self._peers[peer.name] = peer
        return peer.name

    def disconnect_peer(self, peer: str) -> None:
        with self._lock:
            metadata = self._peers.pop(peer, None)
        if metadata is not None:
            self._control.disconnect(metadata.control_address)

    def prepare_descriptors(
        self, peer: str | None, regions: Regions
    ) -> PreparedDescriptors:
        table = np.asarray(regions, dtype=np.uint64)
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError(
                "Transfer descriptors must contain address and byte length"
            )
        table = table[:, :3].copy()
        table.flags.writeable = False
        return PreparedDescriptors(peer, table)

    def release_descriptors(self, descriptors: PreparedDescriptors) -> None:
        # Transfers own their selected addresses; the immutable table is GC-owned.
        pass

    def create_transfer(
        self,
        operation: Operation,
        local: PreparedDescriptors,
        local_indices: np.ndarray,
        remote: PreparedDescriptors,
        remote_indices: np.ndarray,
        notif_msg: bytes,
    ) -> MooncakeTransfer:
        if operation not in ("READ", "WRITE"):
            raise ValueError(f"Unsupported Mooncake transfer operation: {operation}")
        if local.peer is not None or remote.peer is None:
            raise ValueError("Transfers require local then remote descriptors")
        local_indices = np.asarray(local_indices, dtype=np.int64)
        remote_indices = np.asarray(remote_indices, dtype=np.int64)
        if (local_indices < 0).any() or (remote_indices < 0).any():
            raise ValueError("Transfer descriptor indices must be nonnegative")
        local_regions = local.regions[local_indices]
        remote_regions = remote.regions[remote_indices]
        if len(local_regions) != len(remote_regions) or not np.array_equal(
            local_regions[:, 1], remote_regions[:, 1]
        ):
            raise ValueError("Local and remote transfer byte lengths must match")
        with self._lock:
            peer = self._peers[remote.peer]
        return MooncakeTransfer(
            operation,
            peer,
            local_regions[:, 0].tolist(),
            remote_regions[:, 0].tolist(),
            local_regions[:, 1].tolist(),
            notif_msg,
        )

    def submit(self, transfer: MooncakeTransfer) -> None:
        with self._lock:
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._closed or transfer.future is not None:
                raise RuntimeError(
                    "Transfer is already submitted or transport is closed"
                )
            transfer.future = self._executor.submit(self._execute, transfer)
            self._active.add(transfer)

    def _execute(self, transfer: MooncakeTransfer) -> TransferTelemetry:
        start = time.perf_counter()
        if transfer.lengths:
            operation = (
                self.engine.batch_transfer_sync_read
                if transfer.operation == "READ"
                else self.engine.batch_transfer_sync_write
            )
            try:
                ret = operation(
                    transfer.peer.session,
                    transfer.local_addresses,
                    transfer.remote_addresses,
                    transfer.lengths,
                )
                if ret != 0:
                    raise RuntimeError(f"Mooncake {transfer.operation} failed: {ret}")
            except Exception as exc:
                # Native batch timeouts can return before all DMA has stopped.
                # Do not return a recoverable failure that recycles KV blocks.
                error = FatalTransferError(
                    "Mooncake could not establish transfer completion; "
                    "restart the worker before reusing its KV memory"
                )
                with self._lock:
                    self._fatal_error = error
                raise error from exc
        data_done = time.perf_counter()
        self._control.send(transfer.peer.control_address, transfer.notification)
        return TransferTelemetry(
            data_done - start, 0.0, sum(transfer.lengths), len(transfer.lengths)
        )

    def poll(self, transfer: MooncakeTransfer) -> TransferState:
        if self._fatal_error is not None:
            raise self._fatal_error
        if transfer.future is None:
            raise RuntimeError("Transfer has not been submitted")
        if not transfer.future.done():
            return TransferState.PENDING
        return (
            TransferState.FAILED
            if transfer.future.exception() is not None
            else TransferState.DONE
        )

    def telemetry(self, transfer: MooncakeTransfer) -> TransferTelemetry:
        if transfer.future is None or not transfer.future.done():
            raise RuntimeError("Transfer telemetry is not ready")
        return transfer.future.result()

    def release_transfer(self, transfer: MooncakeTransfer) -> None:
        if transfer.future is not None:
            # Do not let shutdown/release unpin buffers with queued DMA work.
            try:
                transfer.future.result()
            except FatalTransferError:
                raise
            except Exception:
                # Notification failure is safe once data movement completed.
                pass
        with self._lock:
            self._active.discard(transfer)

    def send_notification(self, peer: str, notif_msg: bytes) -> None:
        with self._lock:
            address = self._peers[peer].control_address
        self._control.send(address, notif_msg)

    def get_notifications(self) -> dict[str, list[bytes]]:
        return self._control.receive()

    def drain(self) -> None:
        with self._lock:
            active = list(self._active)
        for transfer in active:
            self.release_transfer(transfer)
        if self._fatal_error is not None:
            raise self._fatal_error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)
        try:
            self.drain()
        finally:
            self._control.close()
