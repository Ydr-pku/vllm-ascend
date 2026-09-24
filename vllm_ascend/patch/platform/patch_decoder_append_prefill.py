# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Advertise Scheduler admission capability without another feature switch."""

from fastapi import Request
from vllm.entrypoints.serve.instrumentator.health import router


@router.get("/v1/decoder_append_prefill/capabilities")
async def decoder_append_prefill_capabilities(request: Request) -> dict:
    # Platform loads its patches during initialization: defer this circular import.
    from vllm_ascend.platform import _decoder_append_prefill_capabilities

    return _decoder_append_prefill_capabilities(request.app.state.vllm_config)
