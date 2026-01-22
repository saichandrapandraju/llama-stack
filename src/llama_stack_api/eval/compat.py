# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Backward compatibility helpers for the Eval API.

This module provides utilities to support both the old-style (individual parameters)
and new-style (request objects) calling conventions for Eval API methods.

The old-style parameters are deprecated and will be removed in a future release.
"""

import warnings
from typing import Any

from .models import (
    BenchmarkConfig,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    RunEvalRequest,
)

_DEPRECATION_MESSAGE = (
    "Passing individual parameters to {method_name}() is deprecated. "
    "Please use {request_class}(benchmark_id=..., ...) instead. "
    "This will be removed in a future release."
)


def _emit_deprecation_warning(method_name: str, request_class: str) -> None:
    """Emit a deprecation warning for old-style parameter usage."""
    warnings.warn(
        _DEPRECATION_MESSAGE.format(method_name=method_name, request_class=request_class),
        DeprecationWarning,
        stacklevel=2,
    )


def resolve_run_eval_request(
    request: RunEvalRequest | None = None,
    *,
    benchmark_id: str | None = None,
    benchmark_config: BenchmarkConfig | None = None,
) -> RunEvalRequest:
    """
    Resolve run_eval parameters to a RunEvalRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        benchmark_config: (Deprecated) The benchmark configuration

    Returns:
        RunEvalRequest object
    """
    if request is not None:
        return request

    # Old-style parameters
    if benchmark_id is not None and benchmark_config is not None:
        _emit_deprecation_warning("run_eval", "RunEvalRequest")
        return RunEvalRequest(
            benchmark_id=benchmark_id,
            benchmark_config=benchmark_config,
        )

    raise ValueError("Either 'request' (RunEvalRequest) or both 'benchmark_id' and 'benchmark_config' must be provided")


def resolve_evaluate_rows_request(
    request: EvaluateRowsRequest | None = None,
    *,
    benchmark_id: str | None = None,
    input_rows: list[dict[str, Any]] | None = None,
    scoring_functions: list[str] | None = None,
    benchmark_config: BenchmarkConfig | None = None,
) -> EvaluateRowsRequest:
    """
    Resolve evaluate_rows parameters to an EvaluateRowsRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        input_rows: (Deprecated) The rows to evaluate
        scoring_functions: (Deprecated) The scoring functions to use
        benchmark_config: (Deprecated) The benchmark configuration

    Returns:
        EvaluateRowsRequest object
    """
    if request is not None:
        return request

    # Old-style parameters
    if (
        benchmark_id is not None
        and input_rows is not None
        and scoring_functions is not None
        and benchmark_config is not None
    ):
        _emit_deprecation_warning("evaluate_rows", "EvaluateRowsRequest")
        return EvaluateRowsRequest(
            benchmark_id=benchmark_id,
            input_rows=input_rows,
            scoring_functions=scoring_functions,
            benchmark_config=benchmark_config,
        )

    raise ValueError(
        "Either 'request' (EvaluateRowsRequest) or all of 'benchmark_id', 'input_rows', "
        "'scoring_functions', and 'benchmark_config' must be provided"
    )


def resolve_job_status_request(
    request: JobStatusRequest | None = None,
    *,
    benchmark_id: str | None = None,
    job_id: str | None = None,
) -> JobStatusRequest:
    """
    Resolve job_status parameters to a JobStatusRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        job_id: (Deprecated) The job ID

    Returns:
        JobStatusRequest object
    """
    if request is not None:
        return request

    # Old-style parameters
    if benchmark_id is not None and job_id is not None:
        _emit_deprecation_warning("job_status", "JobStatusRequest")
        return JobStatusRequest(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )

    raise ValueError("Either 'request' (JobStatusRequest) or both 'benchmark_id' and 'job_id' must be provided")


def resolve_job_cancel_request(
    request: JobCancelRequest | None = None,
    *,
    benchmark_id: str | None = None,
    job_id: str | None = None,
) -> JobCancelRequest:
    """
    Resolve job_cancel parameters to a JobCancelRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        job_id: (Deprecated) The job ID

    Returns:
        JobCancelRequest object
    """
    if request is not None:
        return request

    # Old-style parameters
    if benchmark_id is not None and job_id is not None:
        _emit_deprecation_warning("job_cancel", "JobCancelRequest")
        return JobCancelRequest(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )

    raise ValueError("Either 'request' (JobCancelRequest) or both 'benchmark_id' and 'job_id' must be provided")


def resolve_job_result_request(
    request: JobResultRequest | None = None,
    *,
    benchmark_id: str | None = None,
    job_id: str | None = None,
) -> JobResultRequest:
    """
    Resolve job_result parameters to a JobResultRequest object.

    Supports both new-style (request object) and old-style (individual parameters).
    Old-style usage emits a DeprecationWarning.

    Args:
        request: The new-style request object (preferred)
        benchmark_id: (Deprecated) The benchmark ID
        job_id: (Deprecated) The job ID

    Returns:
        JobResultRequest object
    """
    if request is not None:
        return request

    # Old-style parameters
    if benchmark_id is not None and job_id is not None:
        _emit_deprecation_warning("job_result", "JobResultRequest")
        return JobResultRequest(
            benchmark_id=benchmark_id,
            job_id=job_id,
        )

    raise ValueError("Either 'request' (JobResultRequest) or both 'benchmark_id' and 'job_id' must be provided")
