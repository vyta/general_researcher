"""
OpenTelemetry tracing setup for Azure AI Foundry.

Initializes auto-instrumentation for all OpenAI calls so they emit spans
to Application Insights. Call `setup_tracing()` once at application startup
before any LLM calls are made.

Requires APPLICATIONINSIGHTS_CONNECTION_STRING env var to be set explicitly.
"""
import os


def setup_tracing() -> bool:
    """
    Configure OpenTelemetry tracing to Application Insights.

    Returns True if tracing was enabled, False if skipped.
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        return False

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

        configure_azure_monitor(connection_string=connection_string)
        OpenAIInstrumentor().instrument()

        # Enable prompt/response content capture if requested
        if os.getenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "").lower() == "true":
            print("📡 Tracing enabled (with prompt/response content recording)")
        else:
            print("📡 Tracing enabled (set AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true for content capture)")

        return True

    except ImportError as e:
        print(f"⚠️  Tracing packages not installed, skipping: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Tracing setup failed, continuing without tracing: {e}")
        return False
