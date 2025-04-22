from typing import List

from llama_stack.providers.datatypes import (
    Api,
    remote_provider_spec,
    ProviderSpec,
    AdapterSpec
)


def available_providers() -> List[ProviderSpec]:
    return [
        remote_provider_spec(
            api=Api.explanation,
            adapter=AdapterSpec(
                adapter_type='captum',
                module="llama_stack.providers.remote.explanation.captum",
                config_class="llama_stack.providers.remote.explanation.captum.CaptumExplanationConfig",
                pip_packages=[
                    "git+https://github.com/saichandrapandraju/captum@remote-logprobs#egg=captum[remote]",
                ],
            )
        ),
    ]
