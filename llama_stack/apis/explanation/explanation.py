from typing import Optional, Protocol, runtime_checkable, List, Dict, Union, Tuple

from enum import Enum

from pydantic import BaseModel

from llama_stack.apis.inference import InterleavedContent

from llama_stack.schema_utils import json_schema_type, webmethod

@json_schema_type
class TextTemplateInput(BaseModel):
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

# class ExplanationAlgo(Enum):
#     feature_ablation: "fa"
#     shapley_values: "shap"
#     shapley_value_sampling: "shap_sampling"

@runtime_checkable
class Explanation(Protocol):
    @webmethod(route="/explanation", method="POST")
    async def explain(
        self,
        model_id: str,
        content: Union[InterleavedContent, TextTemplateInput],
        algorithm: str,
        target: Optional[InterleavedContent] = None,
        skip_tokens: Optional[List[Union[int, str]]] = None,
        num_trials: Optional[int] = 1,
        gen_args: Optional[Dict[str, Union[str, int]]] = None,
    ) -> ExplanationResponse: ...
