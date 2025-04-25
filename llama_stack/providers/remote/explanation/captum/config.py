from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class CaptumExplanationConfig(BaseModel):
    llms: List[str] = Field(
        default=[],
        description="The URLs for the remote model serving endpoints",
    )
    tokenizers: List[str] = Field(
        default=[],
        description="The HuggingFace tokenizer names for the remote model serving endpoints",
    )
    @field_validator("llms", "tokenizers", mode="before")
    @classmethod
    def validate_llms_tokenizers(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @classmethod
    def sample_run_config(
        cls,
        llms: str = "${env.VLLM_URL}",
        tokenizers: str = "${env.TOKENIZER}",
        **kwargs,
    ) -> Dict[str, Any]:
        
        return {
            "llms": llms,
            "tokenizers": tokenizers,
            **kwargs,
        }
