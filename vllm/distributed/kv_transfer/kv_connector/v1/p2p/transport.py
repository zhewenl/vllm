# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer-engine boundary for the shared P/D protocol.

Descriptors name byte ranges, not KV groups, layers, requests or TP ranks.
Registration and prepared descriptors are persistent; transfers select indices
from prepared descriptors without rebuilding the full table on the hot path.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from vllm.config import VllmConfig

Operation = Literal["READ", "WRITE"]
Handle = Any
Regions: TypeAlias = list[tuple[int, int, int]] | np.ndarray


class TransferState(str, Enum):
    PENDING = "PROC"
    DONE = "DONE"
    FAILED = "FAILED"


class FatalTransferError(RuntimeError):
    """The engine cannot guarantee DMA has stopped; restart the worker.

    This must not be converted to a recoverable request failure or permit
    deregistration/reuse of the affected memory.
    """


@dataclass(frozen=True)
class TransferTelemetry:
    duration_seconds: float
    post_seconds: float
    bytes_transferred: int
    descriptor_count: int


class TransferTransport(ABC):
    """A worker-local engine supporting asynchronous byte-range transfers.

    Both READ and WRITE use local descriptors first, remote descriptors second.
    READ writes local memory; WRITE reads it. DONE includes the engine's data
    visibility guarantee. Notifications must never precede data visibility;
    they may arrive before or after the initiator observes DONE. Notification
    payloads are opaque bytes and each successful transfer emits one message.

    FAILED is terminal with no outstanding memory accesses for that transfer.
    An engine unable to establish this must raise FatalTransferError instead.
    drain() quiesces all outstanding accesses before unregistering memory.
    Implementations must tolerate calls from the handshake and progress threads.
    """

    operations: frozenset[Operation] = frozenset(("READ", "WRITE"))

    @abstractmethod
    def register_memory(self, regions: Regions) -> Handle: ...

    @abstractmethod
    def unregister_memory(self, registration: Handle) -> None: ...

    @abstractmethod
    def export_peer(self) -> bytes: ...

    @abstractmethod
    def connect_peer(self, metadata: bytes) -> str: ...

    @abstractmethod
    def disconnect_peer(self, peer: str) -> None: ...

    @abstractmethod
    def prepare_descriptors(self, peer: str | None, regions: Regions) -> Handle:
        """Prepare a descriptor table; peer=None denotes local memory."""
        ...

    @abstractmethod
    def release_descriptors(self, descriptors: Handle) -> None: ...

    @abstractmethod
    def create_transfer(
        self,
        operation: Operation,
        local: Handle,
        local_indices: np.ndarray,
        remote: Handle,
        remote_indices: np.ndarray,
        notif_msg: bytes,
    ) -> Handle: ...

    @abstractmethod
    def submit(self, transfer: Handle) -> None: ...

    @abstractmethod
    def poll(self, transfer: Handle) -> TransferState:
        """Return immediately; do not wait for the engine to finish."""
        ...

    @abstractmethod
    def telemetry(self, transfer: Handle) -> TransferTelemetry: ...

    @abstractmethod
    def release_transfer(self, transfer: Handle) -> None: ...

    @abstractmethod
    def send_notification(self, peer: str, notif_msg: bytes) -> None: ...

    @abstractmethod
    def get_notifications(self) -> dict[str, list[bytes]]: ...

    @abstractmethod
    def drain(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def get_transfer_engine(config: "VllmConfig") -> str:
    assert config.kv_transfer_config is not None
    connector = config.kv_transfer_config.kv_connector
    default = (
        "mooncake"
        if connector in ("MooncakePullConnector", "MooncakePushConnector")
        else "nixl"
    )
    return config.kv_transfer_config.get_from_extra_config("transfer_engine", default)


_TRANSPORTS = {
    "nixl": (__package__ + ".transports.nixl", "NixlTransport"),
    "mooncake": (__package__ + ".transports.mooncake", "MooncakeTransport"),
}


def register_transport(name: str, module_path: str, class_name: str) -> None:
    """Register an out-of-tree engine without importing it on scheduler ranks."""
    if name in _TRANSPORTS:
        raise ValueError(f"Transfer engine {name!r} is already registered")
    _TRANSPORTS[name] = (module_path, class_name)


def create_transport(
    config: "VllmConfig", operation: Operation = "READ"
) -> TransferTransport:
    engine = get_transfer_engine(config)
    if engine not in _TRANSPORTS:
        raise ValueError(f"Unsupported P/D transfer engine: {engine!r}")
    module_path, class_name = _TRANSPORTS[engine]
    transport_class = getattr(importlib.import_module(module_path), class_name)
    if not issubclass(transport_class, TransferTransport):
        raise TypeError(f"{class_name} must implement TransferTransport")
    if operation not in transport_class.operations:
        raise ValueError(f"Transfer engine {engine!r} does not support {operation}")
    return transport_class(config)
