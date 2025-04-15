from typing import List

from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec
)


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.explanation,
            provider_type="inline::captum",
            pip_packages=[
                "git+https://github.com/saichandrapandraju/captum@remote-logprobs#egg=captum[remote]",
            ],
            module="llama_stack.providers.inline.explanation.captum_explanation",
            config_class="llama_stack.providers.inline.explanation.captum_explanation.CaptumExplanationConfig",
        ),
    ]
