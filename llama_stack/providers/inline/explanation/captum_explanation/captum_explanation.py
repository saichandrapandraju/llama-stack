import logging

from typing import Union, Optional, List, Dict

from llama_stack.apis.explanation import (
    Explanation,
    ExplanationResponse,
    TextTemplateInput,
    # ExplanationAlgo
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
    TextTemplateInput as captum_template_inp,
)
from torch.nn import Module
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

class CaptumExplanationImpl(
    Explanation
):
    def __init__(self, config: CaptumExplanationConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.remote_llm_attr_fa = None
        self.remote_llm_attr_shap = None
        self.remote_llm_attr_shap_sampling = None

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


    async def explain(
            self,
            model_id:str,
            content: Union[InterleavedContent, TextTemplateInput],
            algorithm: str,
            target: Optional[InterleavedContent] = None,
            skip_tokens: Optional[List[Union[int, str]]] = None,
            num_trials: Optional[int] = 1,
            gen_args: Optional[Dict[str, Union[str, int]]] = None,
            ) -> ExplanationResponse:
        
        assert self.tokenizer is not None

        if isinstance(content, TextTemplateInput):
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
                
            inp = captum_template_inp(
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
    