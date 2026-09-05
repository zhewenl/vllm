# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte movement and completion contracts below the shared P/D planner."""

import contextlib
import ctypes
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.transport import (
    FatalTransferError,
    TransferState,
    create_transport,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.transports.control import (
    ControlChannel,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.transports.mooncake import (
    MooncakeTransport,
)


class MemoryEngine:
    """Execute actual byte copies while controlling completion and failures."""

    def __init__(self):
        self.ready = threading.Event()
        self.ready.set()
        self.started = threading.Event()
        self.failure = False

    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 12345

    def batch_register_memory(self, addresses, lengths):
        return 0

    def batch_unregister_memory(self, addresses):
        return 0

    def _transfer(self, local, remote, lengths, read):
        self.started.set()
        assert self.ready.wait(timeout=5)
        if self.failure:
            return -1
        for local_address, remote_address, length in zip(local, remote, lengths):
            source, destination = (
                (remote_address, local_address)
                if read
                else (local_address, remote_address)
            )
            ctypes.memmove(destination, source, length)
        return 0

    def batch_transfer_sync_read(self, peer, local, remote, lengths):
        return self._transfer(local, remote, lengths, True)

    def batch_transfer_sync_write(self, peer, local, remote, lengths):
        return self._transfer(local, remote, lengths, False)


def transport_config(engine="mooncake", **extra):
    options = {"transfer_engine": engine, "mooncake_hostname": "127.0.0.1", **extra}
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="P2pConnector",
            kv_buffer_device="cpu",
            get_from_extra_config=lambda key, default=None: options.get(key, default),
        )
    )


@pytest.fixture
def transports(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(device_type="cpu")
    )
    monkeypatch.setitem(
        sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=MemoryEngine)
    )
    first = create_transport(transport_config())
    second = create_transport(transport_config())
    assert isinstance(first, MooncakeTransport)
    first.connect_peer(second.export_peer())
    second.connect_peer(first.export_peer())
    yield first, second
    first.engine.ready.set()
    second.engine.ready.set()
    with contextlib.suppress(FatalTransferError):
        first.close()
    second.close()


def wait_done(transport, transfer):
    deadline = time.monotonic() + 5
    while transport.poll(transfer) == TransferState.PENDING:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    return transport.poll(transfer)


@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_scattered_transfer_copies_selected_regions_before_notifying(
    transports, operation
):
    """The same plan must preserve holes, index order and exact byte coverage."""
    first, second = transports
    local = ctypes.create_string_buffer(b"abcdefghijklmnop", 16)
    remote = ctypes.create_string_buffer(b"ABCDEFGHIJKLMNOP", 16)
    first_registration = first.register_memory([(ctypes.addressof(local), 16, 0)])
    second_registration = second.register_memory([(ctypes.addressof(remote), 16, 0)])
    local_desc = first.prepare_descriptors(
        None, [(ctypes.addressof(local) + i, 4, 0) for i in range(0, 16, 4)]
    )
    remote_desc = first.prepare_descriptors(
        second._peer.name,
        [(ctypes.addressof(remote) + i, 4, 0) for i in range(0, 16, 4)],
    )
    first.engine.ready.clear()
    transfer = first.create_transfer(
        operation,
        local_desc,
        np.array([2, 0]),
        remote_desc,
        np.array([0, 3]),
        notif_msg=b"request:2",
    )
    first.submit(transfer)
    assert first.engine.started.wait(timeout=5)
    assert first.poll(transfer) == TransferState.PENDING
    assert second.get_notifications() == {}
    first.engine.ready.set()
    assert wait_done(first, transfer) == TransferState.DONE
    if operation == "READ":
        assert local.raw == b"MNOPefghABCDmnop"
        assert remote.raw == b"ABCDEFGHIJKLMNOP"
    else:
        assert local.raw == b"abcdefghijklmnop"
        assert remote.raw == b"ijklEFGHIJKLabcd"
    assert second.get_notifications() == {first._peer.name: [b"request:2"]}
    assert second.get_notifications() == {}
    telemetry = first.telemetry(transfer)
    assert telemetry.bytes_transferred == 8
    assert telemetry.descriptor_count == 2
    first.release_transfer(transfer)
    first.unregister_memory(first_registration)
    second.unregister_memory(second_registration)


@pytest.mark.parametrize("operation", ["READ", "WRITE"])
def test_failed_transfer_never_notifies_success(transports, operation):
    first, second = transports
    first.engine.failure = True
    local = first.prepare_descriptors(None, [(1, 8, 0)])
    remote = first.prepare_descriptors(second._peer.name, [(2, 8, 0)])
    transfer = first.create_transfer(
        operation, local, np.array([0]), remote, np.array([0]), b"done"
    )
    first.submit(transfer)
    with pytest.raises(FatalTransferError, match="restart the worker"):
        wait_done(first, transfer)
    assert second.get_notifications() == {}
    with pytest.raises(FatalTransferError):
        first.drain()
    with pytest.raises(FatalTransferError):
        first.unregister_memory(([1], [8]))


def test_release_waits_for_active_memory_access(transports):
    first, second = transports
    first.engine.ready.clear()
    local_buffer = ctypes.create_string_buffer(8)
    remote_buffer = ctypes.create_string_buffer(b"ABCDEFGH", 8)
    local = first.prepare_descriptors(None, [(ctypes.addressof(local_buffer), 8, 0)])
    remote = first.prepare_descriptors(
        second._peer.name, [(ctypes.addressof(remote_buffer), 8, 0)]
    )
    transfer = first.create_transfer(
        "READ", local, np.array([0]), remote, np.array([0]), b"done"
    )
    first.submit(transfer)
    assert first.engine.started.wait(timeout=5)
    with ThreadPoolExecutor(1) as executor:
        released = executor.submit(first.release_transfer, transfer)
        assert not released.done()
        first.engine.ready.set()
        released.result(timeout=5)
    assert first.poll(transfer) == TransferState.DONE
    assert local_buffer.raw == b"ABCDEFGH"


def test_notification_failure_is_recoverable_after_data_completion(
    transports, monkeypatch
):
    first, second = transports
    local_buffer = ctypes.create_string_buffer(8)
    remote_buffer = ctypes.create_string_buffer(b"ABCDEFGH", 8)
    local = first.prepare_descriptors(None, [(ctypes.addressof(local_buffer), 8, 0)])
    remote = first.prepare_descriptors(
        second._peer.name, [(ctypes.addressof(remote_buffer), 8, 0)]
    )

    def fail_notification(*args):
        raise TimeoutError("peer unreachable")

    monkeypatch.setattr(first._control, "send", fail_notification)
    transfer = first.create_transfer("READ", local, [0], remote, [0], b"done")
    first.submit(transfer)
    assert wait_done(first, transfer) == TransferState.FAILED
    first.release_transfer(transfer)
    first.drain()
    assert local_buffer.raw == b"ABCDEFGH"
    assert second.get_notifications() == {}


def test_reject_mismatched_descriptors_before_submission(transports):
    first, second = transports
    local = first.prepare_descriptors(None, [(1, 8, 0)])
    remote = first.prepare_descriptors(second._peer.name, [(2, 4, 0)])
    with pytest.raises(ValueError, match="byte lengths"):
        first.create_transfer("READ", local, [0], remote, [0], b"done")
    assert not first.engine.started.is_set()


def test_full_hit_notification_needs_no_data_transfer(transports):
    first, second = transports
    first.send_notification(second._peer.name, b"full-hit:1")
    assert second.get_notifications() == {first._peer.name: [b"full-hit:1"]}
    assert not first.engine.started.is_set()


def test_peer_version_rejected_before_connect(transports):
    first, second = transports
    data = msgspec.msgpack.decode(second.export_peer())
    data["version"] = 2
    with pytest.raises(ValueError, match="version"):
        first.connect_peer(msgspec.msgpack.encode(data))


def test_control_channel_drains_concurrent_binary_messages():
    first = ControlChannel("127.0.0.1")
    second = ControlChannel("127.0.0.1")
    messages = [b"PUSH_REG:\x00\xff" + bytes([i]) for i in range(16)]
    try:
        with ThreadPoolExecutor(4) as executor:
            list(executor.map(lambda m: first.send(second.address, m), messages))
        assert sorted(second.receive()[first.name]) == sorted(messages)
        assert second.receive() == {}
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("engine", ["nixl", "mooncake"])
def test_prometheus_reports_engine_metrics_in_seconds(engine):
    """Both backends publish the shared success, failure and lease metrics."""
    from functools import partial

    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.stats import (
        P2pKVConnectorStats,
        P2pPromMetrics,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.transport import (
        TransferTelemetry,
    )

    registry = CollectorRegistry()
    metrics = P2pPromMetrics(
        transport_config(engine),
        {
            metric: partial(metric, registry=registry)
            for metric in (Gauge, Counter, Histogram)
        },
        ["model_name"],
        {0: ["test"]},
    )
    stats = P2pKVConnectorStats()
    stats.record_transfer(TransferTelemetry(0.025, 0.003, 256, 2))
    stats.record_failed_transfer()
    stats.record_failed_notification()
    stats.record_kv_expired_req()
    metrics.observe(stats.clone_and_reset().data)
    assert stats.is_empty()
    for suffix, value in {
        "xfer_time_seconds_sum": 0.025,
        "post_time_seconds_sum": 0.003,
        "bytes_transferred_sum": 256,
        "num_descriptors_sum": 2,
        "num_failed_transfers_total": 1,
        "num_failed_notifications_total": 1,
        "num_kv_expired_reqs_total": 1,
    }.items():
        assert registry.get_sample_value(
            f"vllm:{engine}_{suffix}", {"model_name": "test"}
        ) == pytest.approx(value)


def test_nixl_telemetry_converts_microseconds_at_engine_boundary(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.transports import nixl

    native = MagicMock()
    native.get_xfer_telemetry.return_value = SimpleNamespace(
        xferDuration=25000, postDuration=3000, totalBytes=256, descCount=2
    )
    monkeypatch.setattr(nixl, "NixlWrapper", lambda *args: native)
    monkeypatch.setattr(nixl, "nixl_agent_config", None)
    monkeypatch.setattr(
        nixl,
        "current_platform",
        SimpleNamespace(
            device_type="cpu",
            get_nixl_supported_devices=lambda: {},
            get_nixl_memory_type=lambda: None,
        ),
    )
    transport = create_transport(transport_config("nixl"))
    telemetry = transport.telemetry(1)
    assert telemetry.duration_seconds == 0.025
    assert telemetry.post_seconds == 0.003
    assert telemetry.bytes_transferred == 256
    assert telemetry.descriptor_count == 2
    transport.register_memory([(1024, 256, 0)])
    native.get_reg_descs.assert_called_once_with([(1024, 256, 0, "")], "DRAM")
    transport.close()


def test_unsafe_engine_failure_cannot_recycle_request_blocks():
    from unittest.mock import MagicMock

    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.pull_worker import (
        P2pPullConnectorWorker,
    )

    worker = object.__new__(P2pPullConnectorWorker)
    worker.transport = MagicMock()
    worker.transport.poll.side_effect = FatalTransferError("DMA may still be active")
    worker._handle_failed_transfer = MagicMock()
    transfers = {"request": [object()]}
    with pytest.raises(FatalTransferError):
        worker._pop_done_transfers(transfers)
    assert "request" in transfers
    worker._handle_failed_transfer.assert_not_called()
    worker.transport.release_transfer.assert_not_called()


def test_handshake_rejects_different_engines_and_transfer_directions():
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.metadata import (
        compute_p2p_compatibility_hash,
    )

    from .utils import create_vllm_config

    config = create_vllm_config()
    nixl_pull = compute_p2p_compatibility_hash(config, "FLASH_ATTN")
    config.kv_transfer_config.kv_connector_extra_config["transfer_engine"] = "mooncake"
    mooncake_pull = compute_p2p_compatibility_hash(config, "FLASH_ATTN")
    mooncake_push = compute_p2p_compatibility_hash(config, "FLASH_ATTN", "push")
    assert len({nixl_pull, mooncake_pull, mooncake_push}) == 3
