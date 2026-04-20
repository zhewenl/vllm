# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)


def test_is_empty_on_fresh_stats():
    stats = MooncakeKVConnectorStats()
    assert stats.is_empty()
    assert stats.num_successful_transfers == 0


def test_record_transfer_and_reduce():
    stats = MooncakeKVConnectorStats()
    # 1 MB transfer in 1 ms -> 1000 MB/s throughput
    stats.record_transfer(duration_s=0.001, total_bytes=1 * 2**20, num_descs=4)
    # 2 MB transfer in 2 ms
    stats.record_transfer(duration_s=0.002, total_bytes=2 * 2**20, num_descs=6)
    assert not stats.is_empty()
    assert stats.num_successful_transfers == 2

    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 2
    # avg = (1 + 2) / 2 = 1.5 ms
    assert reduced["Avg xfer time (ms)"] == 1.5
    assert reduced["Avg MB per transfer"] == 1.5
    # 3 MB total / 3 ms total = 1000 MB/s
    assert reduced["Throughput (MB/s)"] == 1000.0
    assert reduced["Avg number of descriptors"] == 5.0
    assert reduced["Num failed transfers"] == 0
    assert reduced["Num failed recvs"] == 0
    assert reduced["Num expired reqs"] == 0


def test_record_failures_keeps_stats_non_empty():
    stats = MooncakeKVConnectorStats()
    stats.record_failed_transfer()
    stats.record_failed_recv()
    stats.record_expired_req()
    assert not stats.is_empty()

    reduced = stats.reduce()
    # No successful transfers -> latency/throughput all zero, but failure
    # counters still surface.
    assert reduced["Num successful transfers"] == 0
    assert reduced["Num failed transfers"] == 1
    assert reduced["Num failed recvs"] == 1
    assert reduced["Num expired reqs"] == 1


def test_aggregate_sums_observations():
    a = MooncakeKVConnectorStats()
    b = MooncakeKVConnectorStats()
    a.record_transfer(duration_s=0.001, total_bytes=1 * 2**20, num_descs=1)
    b.record_transfer(duration_s=0.002, total_bytes=2 * 2**20, num_descs=2)
    b.record_failed_transfer()

    a.aggregate(b)

    assert a.num_successful_transfers == 2
    reduced = a.reduce()
    assert reduced["Num successful transfers"] == 2
    assert reduced["Num failed transfers"] == 1


def test_aggregate_with_empty_other_is_noop():
    a = MooncakeKVConnectorStats()
    a.record_transfer(duration_s=0.001, total_bytes=1, num_descs=1)
    b = MooncakeKVConnectorStats()

    a.aggregate(b)

    assert a.num_successful_transfers == 1


def test_clone_and_reset_hands_off_old_data():
    stats = MooncakeKVConnectorStats()
    stats.record_transfer(duration_s=0.001, total_bytes=1, num_descs=1)
    stats.record_failed_recv()

    snapshot = stats.clone_and_reset()

    assert snapshot.num_successful_transfers == 1
    assert not snapshot.is_empty()
    # Original is now empty.
    assert stats.is_empty()
    assert stats.num_successful_transfers == 0
    # Recording on the original does not mutate the snapshot.
    stats.record_transfer(duration_s=0.005, total_bytes=2, num_descs=2)
    assert snapshot.num_successful_transfers == 1


def test_build_kv_connector_stats_none_returns_empty_instance():
    out = MooncakeConnector.build_kv_connector_stats()
    assert isinstance(out, MooncakeKVConnectorStats)
    assert out.is_empty()


def test_build_kv_connector_stats_with_data_round_trips():
    original = MooncakeKVConnectorStats()
    original.record_transfer(duration_s=0.01, total_bytes=1024, num_descs=3)
    original.record_failed_transfer()

    # Serialized form is the .data dict; build should reconstruct an instance
    # that behaves the same.
    rebuilt = MooncakeConnector.build_kv_connector_stats(data=original.data)

    assert isinstance(rebuilt, MooncakeKVConnectorStats)
    assert rebuilt.num_successful_transfers == 1
    assert rebuilt.reduce()["Num failed transfers"] == 1
