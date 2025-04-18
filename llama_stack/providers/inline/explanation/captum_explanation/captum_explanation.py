import logging
from enum import Enum

from typing import Union, Optional, List, Dict

from llama_stack.apis.explanation import (
    Explanation,
    ExplanationResponse,
    TextTemplateInputSchema,
    ExplanationJob,
    ExplanationJobStatusResponse,
    ExplanationJobResultsResponse,
    ListExplanationJobsResponse
)

from llama_stack.apis.inference import InterleavedContent

from .config import CaptumExplanationConfig

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    ShapleyValueSampling,
    # Lime,
    # KernelShap,
    RemoteLLMAttribution,
    VLLMProvider,
    TextTokenInput,
    TextTemplateInput
)

import uuid

from llama_stack.providers.utils.scheduler import JobArtifact, JobStatus as SchedulerJobStatus, Scheduler
from llama_stack.apis.common.job_types import JobStatus

from torch.nn import Module
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

class ExplanationArtifactType(Enum):
    EXPLANATION = "explanation"

class CaptumExplanationImpl(
    Explanation
):
    def __init__(self, config: CaptumExplanationConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.remote_llm_attr_fa = None
        self.remote_llm_attr_shap = None
        self.remote_llm_attr_shap_sampling = None
        self._scheduler = Scheduler()
    
    async def shutdown(self) -> None:
        await self._scheduler.shutdown()

    async def initialize(self) -> None:
        placeholder_model = Module()
        placeholder_model.device = "cpu"

        attr_fa = FeatureAblation(placeholder_model)
        attr_shap = ShapleyValues(placeholder_model)
        attr_shap_sampling = ShapleyValueSampling(placeholder_model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        log.info(f"Initializing VLLMProvider with base_url={self.config.url}")
        vllm_provider = VLLMProvider(api_url=self.config.url)

        self.remote_llm_attr_fa = RemoteLLMAttribution(attr_method=attr_fa, provider=vllm_provider, tokenizer=self.tokenizer)
        self.remote_llm_attr_shap = RemoteLLMAttribution(attr_method=attr_shap, provider=vllm_provider, tokenizer=self.tokenizer)
        self.remote_llm_attr_shap_sampling = RemoteLLMAttribution(attr_method=attr_shap_sampling, provider=vllm_provider, tokenizer=self.tokenizer)

    @staticmethod
    def _explanation_to_artifact(explanation: ExplanationResponse) -> JobArtifact:

        return JobArtifact(
            type=ExplanationArtifactType.EXPLANATION.value,
            name="explanation",
            uri=None,
            metadata=explanation.model_dump()
        )

    async def explain(
            self,
            model_id:str,
            content: Union[InterleavedContent, TextTemplateInputSchema],
            algorithm: str,
            target: Optional[str] = None,
            skip_tokens: Optional[List[Union[int, str]]] = None,
            num_trials: Optional[int] = 1,
            gen_args: Optional[Dict[str, Union[str, int]]] = None,
        ) -> ExplanationResponse:
        
        assert self.tokenizer is not None

        if isinstance(content, TextTemplateInputSchema):
            baseline_content = {}
            if content.baselines is not None and isinstance(content.baselines, Dict):
                for _,v in content.baselines.items():
                    assert len(v) == 2
                    key = v[0]
                    value = v[1]
                    if isinstance(key, List):
                        key = tuple(key)
                    # if isinstance[value[0], List]:
                    #     value = list(map(tuple, value))
                    baseline_content[key] = value
                
            inp = TextTemplateInput(
                template = content.template,
                values = content.values,
                baselines = baseline_content if baseline_content else content.baselines,
                mask = content.mask
            )
        else:
            inp = TextTokenInput(
                content,
                tokenizer=self.tokenizer,
                skip_tokens=skip_tokens
            )
        
        attr_method = None
        if algorithm == 'fa':
            attr_method = self.remote_llm_attr_fa
        elif algorithm == 'shap':
            attr_method = self.remote_llm_attr_shap
        elif algorithm == 'shap_sampling':
            attr_method = self.remote_llm_attr_shap_sampling
        else:
            raise ValueError(f"algorithm must be one of (fa, shap, shap_sampling)")
        
        attr_res = attr_method.attribute(inp, target=target, skip_tokens=skip_tokens, gen_args=gen_args, num_trials=num_trials)

        return ExplanationResponse(
            input_features=attr_res.input_tokens,
            output_features=attr_res.output_tokens,
            attribution=attr_res.token_attr.tolist(),
            extra_attributions={"sequence_attribution": attr_res.seq_attr_dict}
        )
    
    async def batch_explain(
        self,
        model_id: str,
        content: List[Union[InterleavedContent, TextTemplateInputSchema]],
        algorithm: str, # common to all items
        target: Optional[List[str]] = None,  # per item
        skip_tokens: Optional[List[Union[int, str]]] = None, # common to all items
        num_trials: Optional[int] = 1, # common to all items
        gen_args: Optional[Dict[str, Union[str, int]]] = None, # common to all items
    ) -> ExplanationJob:
        
        job_uuid = str(uuid.uuid4())

        if target and len(target) != len(content):
            raise ValueError("target must be the same length as content")

        async def handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb):
            try:
                on_log_message_cb(f"Starting batch explanation job with {len(content)} items")
                on_status_change_cb(SchedulerJobStatus.running)

                for i, item in enumerate(content):
                    on_log_message_cb(f"Processing item {i+1}/{len(content)}")
                    explanation = await self.explain(
                        model_id=model_id,
                        content=item,
                        algorithm=algorithm,
                        target=target[i] if target else None,
                        skip_tokens=skip_tokens,
                        num_trials=num_trials,
                        gen_args=gen_args
                    )
                    artifact = self._explanation_to_artifact(explanation)
                    print("done artifact")
                    on_artifact_collected_cb(artifact)
                    print("collected artifact")

                on_status_change_cb(SchedulerJobStatus.completed)
                on_log_message_cb("Batch explanation completed successfully")
            except Exception as e:
                on_log_message_cb(f"Error in batch explanation: {str(e)}")
                on_status_change_cb(SchedulerJobStatus.failed)
                raise

        self._scheduler.schedule("batch_explanation", job_uuid, handler)
        return ExplanationJob(job_uuid=job_uuid)

    async def get_explanation_jobs(self) -> ListExplanationJobsResponse:
        return ListExplanationJobsResponse(
            data=[ExplanationJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )

    async def get_explanation_job_status(self, job_uuid: str) -> ExplanationJobStatusResponse:
        job = self._scheduler.get_job(job_uuid)
        print(job.__dict__)

        match job.status:
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                raise NotImplementedError()

        return ExplanationJobStatusResponse(
            job_uuid=job_uuid,
            status=status,
            scheduled_at=job.scheduled_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )

    async def cancel_explanation_job(self, job_uuid: str) -> None:
        self._scheduler.cancel(job_uuid)

    async def get_explanation_job_artifacts(self, job_uuid: str) -> ExplanationJobResultsResponse:
        job = self._scheduler.get_job(job_uuid)
        explanations = [
            ExplanationResponse(**artifact.metadata)
            for artifact in job.artifacts
            if artifact.type == ExplanationArtifactType.EXPLANATION.value
        ]
        return ExplanationJobResultsResponse(job_uuid=job_uuid, results=explanations)

