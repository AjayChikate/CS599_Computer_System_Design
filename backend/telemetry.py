from __future__ import annotations

import atexit
import logging
import os
import sys
import time
from functools import lru_cache
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "pond-catchment-analysis")
logger = logging.getLogger(__name__)
_configured = False
_root_handler: LoggingHandler | None = None
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None
_logger_provider: LoggerProvider | None = None


def configure_telemetry(app: Any) -> None:
    global _configured, _root_handler, _tracer_provider, _meter_provider, _logger_provider
    if _configured:
        return

    resource = Resource.create({"service.name": SERVICE_NAME})
    otlp_enabled = any(
        os.getenv(name)
        for name in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        )
    )

    _tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter() if otlp_enabled else ConsoleSpanExporter(out=sys.__stdout__)
    _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(_tracer_provider)

    metric_exporter = OTLPMetricExporter() if otlp_enabled else ConsoleMetricExporter(out=sys.__stdout__)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
    _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(_meter_provider)

    _logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter() if otlp_enabled else ConsoleLogRecordExporter(out=sys.__stdout__)
    _logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    _root_handler = LoggingHandler(level=logging.INFO, logger_provider=_logger_provider)
    logging.getLogger().addHandler(_root_handler)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=_tracer_provider)
    _configured = True
    atexit.register(shutdown_telemetry)
    logger.info(
        "OpenTelemetry configured: service=%s exporter=%s",
        SERVICE_NAME,
        "otlp-http" if otlp_enabled else "console",
    )


def shutdown_telemetry() -> None:
    if _root_handler is not None:
        logging.getLogger().removeHandler(_root_handler)
        _root_handler.close()
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
    if _meter_provider is not None:
        _meter_provider.shutdown()
    if _logger_provider is not None:
        _logger_provider.shutdown()


@lru_cache(maxsize=1)
def _http_instruments():
    meter = metrics.get_meter(SERVICE_NAME)
    return (
        meter.create_counter(
            "pond.http.server.requests",
            unit="{request}",
            description="Number of HTTP requests handled by the pond analysis API.",
        ),
        meter.create_histogram(
            "pond.http.server.duration",
            unit="s",
            description="Time spent handling HTTP requests.",
        ),
    )


@lru_cache(maxsize=1)
def _analysis_instruments():
    meter = metrics.get_meter(SERVICE_NAME)
    return (
        meter.create_counter(
            "pond.analysis.runs",
            unit="{run}",
            description="Number of successful pond-analysis operations.",
        ),
        meter.create_histogram(
            "pond.analysis.candidates",
            unit="{candidate}",
            description="Number of pond candidates returned by an analysis operation.",
        ),
    )


def record_http_request(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    request_counter, request_duration = _http_instruments()
    attributes = {
        "http.request.method": method,
        "http.route": route,
        "http.response.status_code": status_code,
    }
    request_counter.add(1, attributes)
    request_duration.record(duration_seconds, attributes)


def record_analysis(operation: str, candidate_count: int) -> None:
    analysis_counter, candidates_histogram = _analysis_instruments()
    attributes = {"pond.analysis.operation": operation}
    analysis_counter.add(1, attributes)
    candidates_histogram.record(candidate_count, attributes)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def elapsed_seconds(started_at: float) -> float:
    return time.perf_counter() - started_at
