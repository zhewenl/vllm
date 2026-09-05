# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL adapter; all NIXL-specific descriptors and telemetry stay here."""

import time
import uuid
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.platforms import current_platform

from ..transport import (
    Handle,
    Operation,
    Regions,
    TransferState,
    TransferTelemetry,
    TransferTransport,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class NixlTransport(TransferTransport):
    def __init__(self, config: "VllmConfig"):
        if NixlWrapper is None:
            raise RuntimeError("NIXL is not available")
        assert (transfer_config := config.kv_transfer_config) is not None
        self.backends = transfer_config.get_from_extra_config("backends", ["UCX"])
        options = None
        if nixl_agent_config is not None:
            if any(b != "UCX" for b in self.backends):
                options = nixl_agent_config(
                    backends=self.backends, capture_telemetry=True
                )
            else:
                # Limit UCX UAR usage, preserving the existing NIXL default.
                options = nixl_agent_config(
                    num_threads=transfer_config.get_from_extra_config("num_threads", 4),
                    capture_telemetry=True,
                )
        supported = {
            "cuda": ("cuda", "cpu"),
            "tpu": ("cpu",),
            "xpu": ("cpu", "xpu"),
            "cpu": ("cpu",),
        }
        supported.update(current_platform.get_nixl_supported_devices())
        buffer_device = transfer_config.kv_buffer_device
        if buffer_device not in supported.get(current_platform.device_type, ()):
            raise ValueError(f"NIXL does not support buffer device {buffer_device!r}")
        self.memory_type = current_platform.get_nixl_memory_type()
        if self.memory_type is None:
            self.memory_type = {"cpu": "DRAM", "cuda": "VRAM", "xpu": "VRAM"}.get(
                buffer_device
            )
        if self.memory_type is None:
            raise ValueError(f"No NIXL memory type for buffer device {buffer_device!r}")
        self.agent = NixlWrapper(str(uuid.uuid4()), options)
        self._active: set[Handle] = set()

    def register_memory(self, regions: Regions) -> Handle:
        registration_regions = [
            (int(address), int(length), int(device), "")
            for address, length, device in regions
        ]
        descriptors = self.agent.get_reg_descs(registration_regions, self.memory_type)
        self.agent.register_memory(descriptors, backends=self.backends)
        return descriptors

    def unregister_memory(self, registration: Handle) -> None:
        self.agent.deregister_memory(registration)

    def export_peer(self) -> bytes:
        return self.agent.get_agent_metadata()

    def connect_peer(self, metadata: bytes) -> str:
        return self.agent.add_remote_agent(metadata)

    def disconnect_peer(self, peer: str) -> None:
        self.agent.remove_remote_agent(peer)

    def prepare_descriptors(self, peer: str | None, regions: Regions) -> Handle:
        descriptors = self.agent.get_xfer_descs(regions, self.memory_type)
        return self.agent.prep_xfer_dlist(peer or "NIXL_INIT_AGENT", descriptors)

    def release_descriptors(self, descriptors: Handle) -> None:
        self.agent.release_dlist_handle(descriptors)

    def create_transfer(
        self,
        operation: Operation,
        local: Handle,
        local_indices: np.ndarray,
        remote: Handle,
        remote_indices: np.ndarray,
        notif_msg: bytes,
    ) -> Handle:
        return self.agent.make_prepped_xfer(
            operation,
            local,
            local_indices,
            remote,
            remote_indices,
            notif_msg=notif_msg,
        )

    def submit(self, transfer: Handle) -> None:
        self.agent.transfer(transfer)
        self._active.add(transfer)

    def poll(self, transfer: Handle) -> TransferState:
        state = self.agent.check_xfer_state(transfer)
        if state == "DONE":
            return TransferState.DONE
        if state == "PROC":
            return TransferState.PENDING
        return TransferState.FAILED

    def telemetry(self, transfer: Handle) -> TransferTelemetry:
        telemetry = self.agent.get_xfer_telemetry(transfer)
        return TransferTelemetry(
            telemetry.xferDuration / 1e6,
            telemetry.postDuration / 1e6,
            telemetry.totalBytes,
            telemetry.descCount,
        )

    def release_transfer(self, transfer: Handle) -> None:
        self.agent.release_xfer_handle(transfer)
        self._active.discard(transfer)

    def send_notification(self, peer: str, notif_msg: bytes) -> None:
        self.agent.send_notif(peer, notif_msg=notif_msg)

    def get_notifications(self) -> dict[str, list[bytes]]:
        return self.agent.get_new_notifs()

    def drain(self) -> None:
        for transfer in list(self._active):
            while self.poll(transfer) == TransferState.PENDING:
                time.sleep(0.001)

    def close(self) -> None:
        self.drain()
