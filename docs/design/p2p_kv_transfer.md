# Transfer-engine-agnostic P/D connectors

The existing `NixlConnector`, `NixlPullConnector` and `NixlPushConnector`
implementations remain in `v1/nixl/`. Their default configuration, imports,
metadata and native NIXL execution path are preserved. No protocol files are
moved or replaced with aliases.

The new `P2pPullConnector` and `P2pPushConnector` are opt-in entry points. They
reuse the existing NIXL scheduler and worker through inheritance, with a small
bridge to an engine-independent `TransferTransport` interface. Native Mooncake
READ and WRITE therefore reuse the same HMA/SSM geometry, TP mapping, prefix
handling, leases and metrics without duplicating that protocol code.

## Boundaries and compatibility

```mermaid
flowchart TD
    Old[Existing NixlConnector entry points] --> R[Existing protocol in v1/nixl]
    New[Opt-in P2p / Mooncake entry points] --> R
    R -->|existing configuration| NX[Native NIXL wrapper]
    R -->|opt-in worker factory| B[TransferAgentBridge]
    B --> T[TransferTransport]
    T --> N[NixlTransport]
    T --> C[MooncakeTransport]
    N --> NX
    C --> MC[Native Mooncake TransferEngine]
```

Changes to existing NIXL code are limited to worker factory/hash extension
hooks, propagation of an unsafe-transfer error, and an optional metrics prefix
whose default remains `nixl`. Existing scheduling, geometry and lifecycle code
stays in its original files, and existing NIXL tests keep their original imports
and mocks.

`p2p/legacy.py` translates the existing wrapper method names, registration
metadata, local-agent sentinel and telemetry units to `TransferTransport`.
This bridge is only used by the new entry points. The transport itself knows
byte ranges, opaque peer metadata and binary notifications; it does not know
requests, layers, HMA groups or TP ranks.

This is an incremental migration boundary. The new connectors still inherit
the existing runtime's platform/layout restrictions and NIXL-named internal
helpers. Once native-engine compatibility and performance are validated, a
later change can move the protocol into a neutral directory and remove the
bridge. Such a move is not needed for fixes in the current protocol to be
shared: both paths already execute those same methods.

## Transport contract

The interface lives in `p2p/transport.py`:

1. Register `(address, byte_length, device_id)` memory regions and export peer
   metadata. Connect remote peers using their opaque metadata.
2. Prepare persistent local and remote descriptor tables. Local tables use
   `peer=None`; remote tables use the identifier returned by `connect_peer`.
3. Create a READ or WRITE using indices into the two tables. Local descriptors
   are always first: READ fills local memory, WRITE reads local memory. Paired
   descriptors have equal byte lengths. Creating a handle does not submit work.
4. `submit` starts work; `poll` must return without waiting for data movement.
   A completion notification must never precede visibility of the transferred
   data. Requests own the underlying buffers until protocol completion.
5. Collect normalized telemetry, then release transfer handles. Descriptors and
   registrations outlive every transfer that uses them. Stop submissions and
   drain transfers before releasing registrations at shutdown.

`FAILED` permits request-level recovery only when memory access has stopped.
`FatalTransferError` indicates that the engine cannot establish that guarantee;
this error propagates out of the shared progress loop so the worker is restarted
instead of recycling potentially active KV memory.

NIXL uses prepared native descriptor lists and native completion/notification
operations. Its existing backend options and microsecond-to-second telemetry
conversion remain in `NixlTransport`.

Mooncake executes native `batch_transfer_sync_read` / `batch_transfer_sync_write`
in a bounded worker pool. Submission and polling do not wait for those calls.
A separate ZMQ I/O thread delivers acknowledged binary notifications, including
heartbeats and push registrations. All sockets belong to that thread. A native
batch failure is fatal because Mooncake's timeout path can return without a
quiescence guarantee. A notification failure after successful data movement is
recoverable. Standalone notification sends wait for acknowledgement up to
`notification_timeout`; data-copy polling remains nonblocking.

Out-of-tree transports can implement `TransferTransport` and call
`register_transport(name, module_path, class_name)` during plugin initialization
in each process. The class advertises READ/WRITE capabilities through
`operations`; unsupported directions fail before the engine is constructed.

## Configuration and migration

For native Mooncake pull, configure both instances with this connector. Use
`kv_producer` on prefill and `kv_consumer` on decode:

```json
{
  "kv_connector": "P2pPullConnector",
  "kv_role": "kv_consumer",
  "kv_buffer_device": "cuda",
  "kv_connector_extra_config": {
    "transfer_engine": "mooncake",
    "mooncake_protocol": "rdma"
  }
}
```

For push, change the connector on both instances to `P2pPushConnector`. For NIXL,
set `transfer_engine` to `nixl`. This default preserves existing NIXL deployments.

| Connector name | Direction | Default engine |
| --- | --- | --- |
| `P2pConnector`, `P2pPullConnector` | READ / pull | NIXL |
| `P2pPushConnector` | WRITE / push | NIXL |
| `NixlConnector`, `NixlPullConnector` | READ / pull | NIXL |
| `NixlPushConnector` | WRITE / push | NIXL |
| `MooncakePullConnector` | READ / pull | Mooncake |
| `MooncakePushConnector` | WRITE / push | Mooncake |

The NIXL names and import paths retain their original implementations. Existing
NIXL entry points continue selecting the native wrapper; select a new P2p or
Mooncake entry point to opt in to `transfer_engine`. The default NIXL
compatibility hash retains its existing factors and wire version. Different
engines or transfer directions are rejected during handshake; engine-agnostic
implementation does not imply NIXL-to-Mooncake wire interoperability.

The existing `MooncakeConnector` retains its original implementation and
bootstrap protocol. Migrate both endpoints and the proxy together. Use the
[shared pull proxy](../../examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py)
or [shared push proxy](../../examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py)
with the new connectors, rather than the legacy Mooncake proxy.

Shared `kv_connector_extra_config` options include `side_channel_host` and
`side_channel_port`, falling back to the existing `VLLM_NIXL_SIDE_CHANNEL_*`
variables. The side-channel address must be reachable from the peer. Mooncake
also accepts `mooncake_hostname` (defaults to the worker IP), `mooncake_protocol`
(default `rdma`), `device_name` (default empty), `num_workers` (default 10), and
`notification_timeout` in seconds (default 5). Native Mooncake RPC and control
ports are dynamically allocated and advertised in peer metadata. Deploy peers
on the same trusted network used for vLLM's other distributed communication.
Mooncake requires a build exposing native batch READ and WRITE operations.

## Feature scope and metrics

The shared runtime includes both directions, heterogeneous TP and block sizes,
HMA/SSM and sliding-window descriptor planning, prefix-cache accounting, host
transfer buffers, heterogeneous layout postprocessing, bidirectional pull
transfer, lease heartbeats, peer eviction and the existing push PP protocol.
These reuse the existing NIXL implementation and retain its configuration guards;
sharing code does not expand its supported model/layout/topology combinations.
In particular, push DCP and hybrid SSM DCP remain unsupported, and generic HMA
load-failure reporting still has the existing group-zero limitation. Layerwise
transfer is not implemented by this runtime.

Both backends publish these metrics, with `ENGINE` set to `nixl` or `mooncake`:

- `vllm:ENGINE_xfer_time_seconds`
- `vllm:ENGINE_post_time_seconds`
- `vllm:ENGINE_bytes_transferred`
- `vllm:ENGINE_num_descriptors`
- `vllm:ENGINE_num_failed_transfers_total`
- `vllm:ENGINE_num_failed_notifications_total`
- `vllm:ENGINE_num_kv_expired_reqs_total`

The first four are histograms; the remainder are counters. NIXL metric names
are unchanged. Mooncake measures native batch duration and transferred bytes;
post time is zero because its synchronous API does not report a separate
posting duration. Fatal errors require worker recovery and may terminate the
worker before its last metrics batch is published.

## Validation

`test_p2p_transport.py` checks real byte copies through an in-process native API
stand-in, notification ordering, nonblocking polling, unsafe failures and actual
Prometheus samples. Unmodified NIXL suites exercise the existing protocol and HMA geometry. New
tests check that the native entry points still select the original classes and
wrapper factory, and that the bridge preserves byte movement and telemetry.

Hardware validation is also required: run pull and push with each native
engine, compare deterministic completions against a colocated baseline, run the
existing [P/D accuracy evaluation](../../tests/v1/kv_connector/nixl_integration/test_accuracy.py),
and cover prefix hits, concurrent requests, GQA TP replication boundaries and
hybrid GDN/Mamba models. Mock transport tests cannot establish RDMA/GPU memory
visibility, model accuracy or performance equivalence.
