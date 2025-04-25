from llama_stack.distribution.datatypes import Provider
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings
from llama_stack.providers.remote.explanation.captum import CaptumExplanationConfig

def get_distribution_template() -> DistributionTemplate:
    providers = {
        "explanation": ["remote::captum"],
        "telemetry": ["inline::meta-reference"],
    }
    name = "remote-captum"
    explanation_provider = Provider(
        provider_id="captum-explanation",
        provider_type="remote::captum",
        config=CaptumExplanationConfig.sample_run_config(
            llms="${env.LLMS}",
            tokenizers="${env.TOKENIZERS}"
        ),
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use Captum for explanations.",
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "explanation": [explanation_provider],
                },
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
        },
    )
