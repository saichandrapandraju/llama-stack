from typing import Any, Dict

from .config import CaptumExplanationConfig


async def get_adapter_impl(config: CaptumExplanationConfig, deps: Dict[str, Any]):
    from .captum import CaptumExplanationImpl

    impl = CaptumExplanationImpl(config)
    await impl.initialize()
    return impl