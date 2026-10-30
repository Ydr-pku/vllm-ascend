# Decoder Append-Prefill

Decoder append-prefill reduces repeated Prefiller work in multi-turn agentic
sessions. After the first turn uses normal Prefill/Decode (P/D) disaggregation,
the proxy can send later turns directly to the same Decoder endpoint and its
original data-parallel (DP) rank. If the Decoder still has a usable local KV
prefix, it computes the uncached suffix and generates the response.

Session affinity identifies where KV may be resident. The Decoder Scheduler
checks actual local cache residency when admitting each request. A cache miss
is rejected before scheduling prefill, so the proxy can send the complete
request through normal P/D instead. Affinity does not reserve KV blocks or
guarantee a cache hit.

The feature is opt-in in the Decoder configuration and at the proxy. It is
intended for workloads with growing conversation histories and relatively short
outputs, where repeated Prefiller work can become a
bottleneck. Its benefit depends on cache residency, request lengths, Decoder
load and the cost of cache-miss fallback.

## Requirements

- A working vLLM V1 P/D-disaggregated deployment, with Decode services configured
  as `kv_role="kv_consumer"` and prefix caching enabled.
- `scheduler_config.decoder_append_prefill_config.enabled=true` in every
  Decoder's `--additional-config`. The platform selects the append-prefill
  admission Scheduler for its synchronous or asynchronous scheduling mode.
- The standalone
  `examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py`
  proxy, started with `--enable-decoder-append-prefill`.
- A stable session identifier and the complete conversation history on every
  request. The proxy retains routing affinity, not conversation messages.
- For streaming, an API server that preserves zero-token terminal responses
  from the Scheduler. See [Streaming compatibility](#streaming-compatibility).

## Configure the Decode service

Keep the Prefill service's normal P/D connector configuration, including its
`kv_producer` role. On the Decode service, enable prefix caching, retain the
matching P/D connector with the `kv_consumer` role and add
`decoder_append_prefill_config` under `scheduler_config` in `--additional-config`.
As with DyntraLB, the platform selects the Scheduler from this configuration;
`--scheduler-cls` is not required. Merge these keys into any existing
`--additional-config`, and configure the feature only on Decode nodes.

For example, the following shows a synchronous Decode service using Mooncake
and one internal DP rank:

```bash
vllm serve /path/to/model \
    --host 127.0.0.1 \
    --port 8200 \
    --data-parallel-size 1 \
    --enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{
      "scheduler_config": {
        "decoder_append_prefill_config": {
          "enabled": true
        }
      }
    }' \
    --kv-transfer-config '{
      "kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "28000"
    }'
```

Adapt the model, device allocation, parallelism and network settings to your
existing P/D deployment. The example does not allocate separate devices for
the Prefill and Decode services.

For asynchronous scheduling, replace `--no-async-scheduling` with
`--async-scheduling`; keep the same `--additional-config`. The platform selects
`DecoderAppendPrefillScheduler` for synchronous scheduling and
`AsyncDecoderAppendPrefillScheduler` for asynchronous scheduling.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `scheduler_config.decoder_append_prefill_config.enabled` | bool | `false` | Enables append-prefill admission and selects its Scheduler on the Decode node. |

Choose the scheduling mode supported by your model and deployment. When the
feature is disabled, existing Scheduler selection is unchanged. With no
explicit Scheduler and no other Ascend scheduling policy, vLLM retains its
native synchronous or asynchronous Scheduler.

The backend configuration and proxy flag are independent. Before enabling
append-prefill at the proxy, ensure every Decoder uses `kv_consumer`, prefix
caching and the append-prefill configuration above. The proxy does not configure
remote Decoders, inspect their Scheduler or validate backend support. Enabling
Recompute alone does not enable append-prefill admission; see the combined
configuration below.

## Start the proxy

Install the standalone proxy dependencies in your serving environment:

```bash
python -m pip install httpx fastapi anyio uvicorn
```

From the vLLM Ascend repository root, start the proxy with the addresses of your
running services. This example assumes Prefill on port `8100` and Decode on
port `8200`:

```bash
python examples/disaggregated_prefill_v1/decoder_append_prefill_proxy_server.py \
    --host 127.0.0.1 --port 9000 --workers 1 \
    --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
    --decoder-hosts 127.0.0.1 --decoder-ports 8200 \
    --decoder-data-parallel-size 1 \
    --decoder-engine-epoch decoder-deployment-001 \
    --session-affinity-ttl 3600 \
    --enable-decoder-append-prefill
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--enable-decoder-append-prefill` | Disabled | Enables session-affine Decoder-local routing. Without it, requests use normal P/D. |
| `--decoder-data-parallel-size` | `1` | Internal attention DP size of each Decoder endpoint. It must match every configured Decoder service. |
| `--decoder-engine-epoch` | Random label | Deployment identity for session bindings. Change it and restart the proxy after restarting Decoders. |
| `--session-affinity-ttl` | `3600` seconds | Idle lifetime of a session binding. It must be positive and does not guarantee KV retention. |
| `--workers` | `1` | Number of proxy worker processes. Use one worker for the current validation scope. |

Deployment configuration must guarantee these requirements for every Decoder,
including newly added or restarted services. Recheck the backend role, prefix
caching and Scheduler selection when changing the deployment. The proxy
assumes the configured Decoders support append-prefill admission; it performs
no capability probing at startup or before local dispatch. Model execution and
streaming terminal-event compatibility also require deployment validation.

If the backends require authentication, configure `OPENAI_API_KEY` for the
proxy so it can authenticate its backend requests.

## Send a multi-turn request

Use the same `X-Session-ID` on every turn. Alternatively, provide a nonempty
`conversation_id` string in the JSON body; the header takes precedence if both
are present. No request-mode header or client-supplied cache-admission flag is
needed.

The following non-streaming example discovers the served model, sends the
first turn through P/D, and includes the actual assistant reply in the next
turn's full history:

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:9000", timeout=120.0) as client:
    models = client.get("/v1/models")
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    messages = [{"role": "user", "content": "Explain prefix caching in two sentences."}]

    for turn in range(2):
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Session-ID": "agent-session-001"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 128,
                "stream": False,
            },
        )
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]
        print(reply["content"])
        print(result.get("proxy_metrics", {}))
        messages.append(reply)
        if turn == 0:
            messages.append({"role": "user", "content": "How does eviction affect it?"})
```

The first turn establishes the Decoder owner and DP rank. A subsequent turn
within the affinity lifetime attempts the same owner. Short prompts, changed
tokenized prefixes, eviction or cache resets can still cause a miss and P/D
fallback; the example does not guarantee a resident cache hit. Requests without
a session identifier use P/D, and concurrent requests with the same session
identifier are rejected while that session has an in-flight request.

Use `proxy_metrics.initial_route`, `final_route` and `admission_fallback` to
distinguish attempted local routing from the route that completed the request.
Owner, DP rank and deployment epoch are also reported. For streaming, HTTP
headers describe the initial route and cannot change after streaming starts;
inspect the response's `proxy_metrics` for fallback information.

## Cache misses and streaming compatibility

The proxy marks local attempts with `vllm_xargs.require_decoder_cache_hit=1`.
The Scheduler requires a usable local prefix that saves prefill work, rather
than accepting remote cache availability as a local hit. An initial miss emits
`stop_reason="decoder_cache_miss"` with no generated output and without
scheduling that request's prefill. The proxy then retries through P/D with the
original history and generation budget.

If cache recovery fails after output has already been generated, the Scheduler
uses `decoder_cache_lost_after_output`. The proxy reports an error and does not
replay the request, which would duplicate output.

### Streaming compatibility

For `stream=true`, the API server must forward a zero-token terminal response
with its stop reason before the first output token. A backend that drops this
event cannot support reliable cache-miss fallback. Selecting the append-prefill
Scheduler does not install or verify this API server behavior. Verify the
terminal response and fallback with the exact vLLM version used for deployment
before enabling streaming traffic.

## Optional Recompute and preempt offload

Append-prefill works with ordinary synchronous or asynchronous scheduling and
does not require Recompute or CPU offload. To combine it with Decode-side
Recompute, enable both sibling configurations in the Decode service's
`--additional-config`:

```bash
--additional-config '{
  "scheduler_config": {
    "decoder_append_prefill_config": {
      "enabled": true
    },
    "recompute_scheduler_enable": true
  }
}'
```

The platform automatically selects `DecoderAppendPrefillRecomputeScheduler`
with `--no-async-scheduling`, or `AsyncDecoderAppendPrefillRecomputeScheduler`
with `--async-scheduling`. No `--scheduler-cls` argument is required. These
configurations remain independent: enabling Recompute alone retains its existing
Scheduler selection and does not enable append-prefill.

For CPU preservation and reload of preempted KV, follow
[Preempt Offload Connector](preempt_offload_connector.md) to configure
`MultiConnector` with the P/D connector and `PreemptOffloadConnector`, including
its CPU memory budget and shared-memory requirements. Keep the top-level
`kv_consumer` role. Recompute and the proxy append-prefill switch remain
independent; enabling one does not enable the other.

A preempted request with its own offload recovery state can wait for its store
to complete and restore its KV even if its NPU-local prefix is no longer
resident. This resumes the same request. It does not admit a later session turn
from CPU merely because that session has affinity. After restore, the blocks
participate in normal prefix caching; later turns can reuse them only while
they remain locally resident on the same owner.

If offload cannot be prepared or recovery fails, marked local requests follow
the cache-miss or no-replay-after-output contract above. Ordinary unmarked P/D
requests retain Recompute's existing `recomputed` protocol. Connector failures
configured to fail the request immediately retain their existing error path.

## Support and validation boundaries

- Use P/D-disaggregated Decode nodes with `kv_consumer`. PD-mixed and ordinary
  non-P/D deployments are outside this feature's scope.
- DyntraLB is unsupported because it requires decode-only Decoder work.
  Balance, ShortRequestFirst, profiling-chunk and batch-job policies are also
  not composed with the append-prefill Scheduler. ShortRequestFirst can still
  be configured independently on Prefill nodes.
- The current target Decode configuration uses pipeline parallel size `1`.
  Combinations with profiling-based chunks or larger pipeline parallel sizes
  need separate admission and pipeline validation.
- The Recompute combination disables its first-decode speculative padding
  optimization for all batches because those batches may contain append-prefill.
  It does not disable speculative decoding itself. Model-specific Recompute and
  preempt-offload restrictions still apply; see the offload guide.
- Session TTL and deployment epochs manage routing metadata. They do not pin
  KV, validate backend identity or provide a cache lease. Change the epoch and
  restart the proxy after Decoder restarts. Use one proxy worker; multi-worker
  integration needs separate validation.

CPU regressions cover synchronous/asynchronous admission, cache-miss fallback,
HTTP/SSE handling, cancellation, restore waiting and failure, in-flight output
ordering, mixed-batch padding, and real CPU block-pool reuse after restoration.
Device transfers and model execution are simulated in these tests. They do not
establish NPU transfer correctness, model accuracy, graph behavior or performance.

Measure end-to-end completed requests, cache-hit and fallback rates, latency,
and Prefill/Decode utilization with your actual model and workload. Any reduction
in Prefiller work must be evaluated against the extra Decoder append-prefill
load; this guide does not claim a general throughput gain or stable capacity.
