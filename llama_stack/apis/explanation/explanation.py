from typing import Optional, Protocol, runtime_checkable, List, Dict, Union, Tuple

from enum import Enum

from datetime import datetime

from pydantic import BaseModel, Field

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.common.job_types import JobStatus
from llama_stack.schema_utils import json_schema_type, webmethod

@json_schema_type
class TextTemplateInputSchema(BaseModel):
    template: InterleavedContent
    values: Union[List[InterleavedContent], Dict[str, InterleavedContent]]
    baselines: Optional[Union[List[InterleavedContent], Dict[str, List[InterleavedContent]]]] = None
    mask: Optional[Dict[InterleavedContent, int]] = None

@json_schema_type
class ExplanationResponse(BaseModel):
    input_features: List[InterleavedContent]
    output_features: List[InterleavedContent]
    attribution: List[List[float]]
    extra_attributions: Optional[Dict[str, Dict[InterleavedContent, float]]]

@json_schema_type
class ExplanationJobStatusResponse(BaseModel):
    """Status of an Explanation job."""

    job_uuid: str
    status: JobStatus

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# class ExplanationAlgo(Enum):
#     feature_ablation: "fa"
#     shapley_values: "shap"
#     shapley_value_sampling: "shap_sampling"

class ExplanationJob(BaseModel):
    job_uuid: str

class ListExplanationJobsResponse(BaseModel):
    data: List[ExplanationJob]

@json_schema_type
class ExplanationJobResultsResponse(BaseModel):
    """Artifacts of an explanation job."""

    job_uuid: str
    results: List[ExplanationResponse] = Field(default_factory=list)


@runtime_checkable
class Explanation(Protocol):
    @webmethod(route="/explanation/online", method="POST")
    async def explain(
        self,
        model_id: str,
        content: Union[InterleavedContent, TextTemplateInputSchema],
        algorithm: str,
        target: Optional[str] = None,
        skip_tokens: Optional[List[Union[int, str]]] = None,
        num_trials: Optional[int] = 1,
        gen_args: Optional[Dict[str, Union[str, int]]] = None,
    ) -> ExplanationResponse: ...

    @webmethod(route="/explanation/batch", method="POST")
    async def batch_explain(
        self,
        model_id: str,
        content: List[Union[InterleavedContent, TextTemplateInputSchema]],
        algorithm: str,
        target: Optional[List[str]] = [],
        skip_tokens: Optional[List[Union[int, str]]] = None,
        num_trials: Optional[int] = 1,
        gen_args: Optional[Dict[str, Union[str, int]]] = None,
    ) -> ExplanationJob: ...

    @webmethod(route="/explanation/jobs", method="GET")
    async def get_explanation_jobs(self) -> ListExplanationJobsResponse: ...

    @webmethod(route="/explanation/job/status", method="GET")
    async def get_explanation_job_status(self, job_uuid: str) -> ExplanationJobStatusResponse: ...

    @webmethod(route="/explanation/job/cancel", method="POST")
    async def cancel_explanation_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/explanation/job/artifacts", method="GET")
    async def get_explanation_job_artifacts(self, job_uuid: str) -> ExplanationJobResultsResponse: ...
