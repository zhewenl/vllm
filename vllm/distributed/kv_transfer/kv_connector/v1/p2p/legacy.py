# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary bridge from the existing NIXL runtime to TransferTransport.

Keep protocol logic in its original modules during the opt-in migration. Only
this bridge knows the old wrapper's method names, units and local-agent sentinel.
"""

from types import SimpleNamespace
from typing import Any

from .transport import TransferTransport


class TransferAgentBridge:
    def __init__(self, transport: TransferTransport):
        self.transport = transport
        self._registrations: dict[int, Any] = {}

    def get_reg_descs(self, regions, memory_type):
        return [(int(r[0]), int(r[1]), int(r[2])) for r in regions]

    def register_memory(self, descriptors, backends=None):
        self._registrations[id(descriptors)] = self.transport.register_memory(
            descriptors
        )

    def deregister_memory(self, descriptors):
        registration = self._registrations[id(descriptors)]
        self.transport.unregister_memory(registration)
        del self._registrations[id(descriptors)]

    def get_agent_metadata(self):
        return self.transport.export_peer()

    def add_remote_agent(self, metadata):
        return self.transport.connect_peer(metadata)

    def remove_remote_agent(self, peer):
        self.transport.disconnect_peer(peer)

    def get_xfer_descs(self, regions, memory_type):
        return regions

    def prep_xfer_dlist(self, peer, descriptors):
        return self.transport.prepare_descriptors(
            None if peer == "NIXL_INIT_AGENT" else peer, descriptors
        )

    def release_dlist_handle(self, descriptors):
        self.transport.release_descriptors(descriptors)

    def make_prepped_xfer(
        self, operation, local, local_indices, remote, remote_indices, notif_msg
    ):
        return self.transport.create_transfer(
            operation, local, local_indices, remote, remote_indices, notif_msg
        )

    def transfer(self, handle):
        self.transport.submit(handle)

    def check_xfer_state(self, handle):
        return self.transport.poll(handle).value

    def get_xfer_telemetry(self, handle):
        telemetry = self.transport.telemetry(handle)
        return SimpleNamespace(
            xferDuration=telemetry.duration_seconds * 1e6,
            postDuration=telemetry.post_seconds * 1e6,
            totalBytes=telemetry.bytes_transferred,
            descCount=telemetry.descriptor_count,
        )

    def release_xfer_handle(self, handle):
        self.transport.release_transfer(handle)

    def send_notif(self, peer, notif_msg):
        self.transport.send_notification(peer, notif_msg)

    def get_new_notifs(self):
        return self.transport.get_notifications()
