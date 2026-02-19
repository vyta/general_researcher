"""
OpenTelemetry tracing with dual-mode export.

Supports two exporters simultaneously:
  - In-memory: always active, used by eval framework to assert against spans
  - Cloud (App Insights): opt-in via APPLICATIONINSIGHTS_CONNECTION_STRING

Usage:
    from utils.tracing import setup_tracing, get_span_capture

    setup_tracing()  # call once at startup

    capture = get_span_capture()
    capture.clear()
    # ... run agent ...
    spans = capture.get_finished_spans()
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton
_span_capture = None
_tracer = None


def setup_tracing(enable_cloud: Optional[bool] = None) -> bool:
    """Configure OpenTelemetry tracing.

    Args:
        enable_cloud: Force cloud export on/off. If None, auto-detects
                      from APPLICATIONINSIGHTS_CONNECTION_STRING.

    Returns True if tracing was initialized.
    """
    global _span_capture, _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export import InMemorySpanExporter

        # In-memory exporter — always active
        _span_capture = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_span_capture))

        # Cloud exporter — conditional
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if enable_cloud is True or (enable_cloud is None and connection_string):
            try:
                from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
                cloud_exporter = AzureMonitorTraceExporter(connection_string=connection_string)
                provider.add_span_processor(SimpleSpanProcessor(cloud_exporter))
                logger.info("Tracing: cloud export enabled (App Insights)")
            except Exception as e:
                logger.warning("Tracing: cloud export failed, in-memory only: %s", e)
        else:
            logger.info("Tracing: in-memory only (no APPLICATIONINSIGHTS_CONNECTION_STRING)")

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("general-researcher")

        # Instrument OpenAI calls
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
        except Exception as e:
            logger.debug("OpenAI auto-instrumentation not available: %s", e)

        return True

    except ImportError as e:
        logger.warning("OpenTelemetry not installed, tracing disabled: %s", e)
        return False
    except Exception as e:
        logger.warning("Tracing setup failed: %s", e)
        return False


def get_span_capture():
    """Return the in-memory span exporter for reading captured spans."""
    return _span_capture


def get_tracer():
    """Return the application tracer for creating custom spans."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer("general-researcher")
        except ImportError:
            return None
    return _tracer
