# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for Eval API backward compatibility.

These tests verify that both old-style (individual parameters) and new-style
(request objects) calling conventions work correctly, and that old-style usage
emits appropriate deprecation warnings.
"""

import warnings

import pytest

from llama_stack_api import (
    BenchmarkConfig,
    EvaluateRowsRequest,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    ModelCandidate,
    RunEvalRequest,
    resolve_evaluate_rows_request,
    resolve_job_cancel_request,
    resolve_job_result_request,
    resolve_job_status_request,
    resolve_run_eval_request,
)
from llama_stack_api.inference import SamplingParams, TopPSamplingStrategy


@pytest.fixture
def sample_benchmark_config():
    return BenchmarkConfig(
        eval_candidate=ModelCandidate(
            model="test-model",
            sampling_params=SamplingParams(max_tokens=100, strategy=TopPSamplingStrategy(temperature=0.7)),
        )
    )


class TestResolveRunEvalRequest:
    """Tests for resolve_run_eval_request."""

    def test_new_style_with_request_object(self, sample_benchmark_config):
        """Test that new-style (request object) works without deprecation warning."""
        request = RunEvalRequest(benchmark_id="bench-123", benchmark_config=sample_benchmark_config)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_run_eval_request(request)

            # No deprecation warning should be emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.benchmark_config == sample_benchmark_config

    def test_old_style_with_individual_params(self, sample_benchmark_config):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_run_eval_request(
                benchmark_id="bench-123",
                benchmark_config=sample_benchmark_config,
            )

            # Deprecation warning should be emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "run_eval" in str(deprecation_warnings[0].message)
            assert "RunEvalRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.benchmark_config == sample_benchmark_config

    def test_missing_parameters_raises_error(self, sample_benchmark_config):
        """Test that missing parameters raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_run_eval_request()
        assert "Either 'request'" in str(exc_info.value)

        with pytest.raises(ValueError):
            resolve_run_eval_request(benchmark_id="bench-123")  # missing benchmark_config

        with pytest.raises(ValueError):
            resolve_run_eval_request(benchmark_config=sample_benchmark_config)  # missing benchmark_id


class TestResolveEvaluateRowsRequest:
    """Tests for resolve_evaluate_rows_request."""

    def test_new_style_with_request_object(self, sample_benchmark_config):
        """Test that new-style (request object) works without deprecation warning."""
        request = EvaluateRowsRequest(
            benchmark_id="bench-123",
            input_rows=[{"test": "data"}],
            scoring_functions=["func1"],
            benchmark_config=sample_benchmark_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_evaluate_rows_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.input_rows == [{"test": "data"}]
        assert result.scoring_functions == ["func1"]

    def test_old_style_with_individual_params(self, sample_benchmark_config):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_evaluate_rows_request(
                benchmark_id="bench-123",
                input_rows=[{"test": "data"}],
                scoring_functions=["func1"],
                benchmark_config=sample_benchmark_config,
            )

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "evaluate_rows" in str(deprecation_warnings[0].message)
            assert "EvaluateRowsRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.input_rows == [{"test": "data"}]
        assert result.scoring_functions == ["func1"]

    def test_missing_parameters_raises_error(self, sample_benchmark_config):
        """Test that missing parameters raises ValueError."""
        with pytest.raises(ValueError):
            resolve_evaluate_rows_request()

        with pytest.raises(ValueError):
            resolve_evaluate_rows_request(
                benchmark_id="bench-123",
                input_rows=[{"test": "data"}],
                # missing scoring_functions and benchmark_config
            )


class TestResolveJobStatusRequest:
    """Tests for resolve_job_status_request."""

    def test_new_style_with_request_object(self):
        """Test that new-style (request object) works without deprecation warning."""
        request = JobStatusRequest(benchmark_id="bench-123", job_id="job-456")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_status_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_old_style_with_individual_params(self):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_status_request(benchmark_id="bench-123", job_id="job-456")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "job_status" in str(deprecation_warnings[0].message)
            assert "JobStatusRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_missing_parameters_raises_error(self):
        """Test that missing parameters raises ValueError."""
        with pytest.raises(ValueError):
            resolve_job_status_request()

        with pytest.raises(ValueError):
            resolve_job_status_request(benchmark_id="bench-123")  # missing job_id

        with pytest.raises(ValueError):
            resolve_job_status_request(job_id="job-456")  # missing benchmark_id


class TestResolveJobCancelRequest:
    """Tests for resolve_job_cancel_request."""

    def test_new_style_with_request_object(self):
        """Test that new-style (request object) works without deprecation warning."""
        request = JobCancelRequest(benchmark_id="bench-123", job_id="job-456")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_cancel_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_old_style_with_individual_params(self):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_cancel_request(benchmark_id="bench-123", job_id="job-456")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "job_cancel" in str(deprecation_warnings[0].message)
            assert "JobCancelRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"


class TestResolveJobResultRequest:
    """Tests for resolve_job_result_request."""

    def test_new_style_with_request_object(self):
        """Test that new-style (request object) works without deprecation warning."""
        request = JobResultRequest(benchmark_id="bench-123", job_id="job-456")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_result_request(request)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"

    def test_old_style_with_individual_params(self):
        """Test that old-style (individual parameters) works and emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_job_result_request(benchmark_id="bench-123", job_id="job-456")

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "job_result" in str(deprecation_warnings[0].message)
            assert "JobResultRequest" in str(deprecation_warnings[0].message)

        assert result.benchmark_id == "bench-123"
        assert result.job_id == "job-456"
