from typing import Any, Dict

from .config import CaptumExplanationConfig


async def get_provider_impl(config: CaptumExplanationConfig, deps: Dict[str, Any]):
    from .captum_explanation import CaptumExplanationImpl

    impl = CaptumExplanationImpl(config)
    await impl.initialize()
    return impl