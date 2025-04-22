from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CaptumExplanationConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the remote model serving endpoint",
    )
    tokenizer: Optional[str] = Field(
        default=None,
        description="The tokenizer for the remote model serving endpoint",
    )
    @classmethod
    def sample_run_config(
        cls,
        url: str = "${env.VLLM_URL}",
        tokenizer: str = "${env.TOKENIZER}",
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "url": url,
            "tokenizer": tokenizer,
        }
