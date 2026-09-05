# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in workers reusing the NIXL protocol without replacing its entry points."""

import threading

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

from .legacy import TransferAgentBridge
from .metadata import compute_p2p_compatibility_hash
from .transport import create_transport


class _TransportWorker(NixlBaseConnectorWorker):
    _push_writer_thread: threading.Thread | None

    def _get_wrapper_cls(self):
        return self._create_bridge

    def _create_bridge(self, agent_name, agent_config):
        self.transport = create_transport(
            self.vllm_config, "READ" if self._TRANSFER_MODE == "pull" else "WRITE"
        )
        return TransferAgentBridge(self.transport)

    def _compute_compatibility_hash(self) -> str:
        return compute_p2p_compatibility_hash(
            self.vllm_config, self.backend_name, self._TRANSFER_MODE
        )

    def shutdown(self):
        if getattr(self, "_transport_closed", False) or not hasattr(self, "transport"):
            return
        self._transport_closed = True
        writer = getattr(self, "_push_writer_thread", None)
        if writer is not None:
            assert isinstance(self, NixlPushConnectorWorker)
            self._push_writer_stop.set()
            self._push_writer_wake.set()
            writer.join()
            self._push_writer_thread = None
        executor = getattr(self, "_handshake_initiation_executor", None)
        if executor is None:
            self.transport.close()
            return
        executor.shutdown(wait=True)
        self.transport.drain()
        if self._TRANSFER_MODE == "push" and not hasattr(self, "_push_writer_stop"):
            # Construction failed before the push-specific fields existed.
            NixlBaseConnectorWorker.shutdown(self)
        else:
            super().shutdown()
        self.transport.close()


class P2pPullConnectorWorker(_TransportWorker, NixlPullConnectorWorker):
    pass


class P2pPushConnectorWorker(_TransportWorker, NixlPushConnectorWorker):
    pass
