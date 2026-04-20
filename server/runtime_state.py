#!/usr/bin/python3
# -*- coding: utf-8 -*-

import threading
import time
from configs import API_MAX_CONCURRENCY, logger

request_semaphore = threading.BoundedSemaphore(max(1, API_MAX_CONCURRENCY))

_state_lock = threading.Lock()
_metrics = {
    "started_at": time.time(),
    "total_requests": 0,
    "success_requests": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "plugin_requests": {},
}
_service_state = {
    "ready": False,
    "model_loaded": False,
    "last_error": "",
}


def set_service_state(*, ready=None, model_loaded=None, last_error=None):
    with _state_lock:
        if ready is not None:
            _service_state["ready"] = bool(ready)
        if model_loaded is not None:
            _service_state["model_loaded"] = bool(model_loaded)
        if last_error is not None:
            _service_state["last_error"] = str(last_error)


def get_service_state():
    with _state_lock:
        return dict(_service_state)


def record_request(latency_ms, success, plugin_id):
    with _state_lock:
        _metrics["total_requests"] += 1
        if success:
            _metrics["success_requests"] += 1
        else:
            _metrics["failed_requests"] += 1
        _metrics["total_latency_ms"] += max(0.0, float(latency_ms))
        key = str(plugin_id)
        _metrics["plugin_requests"][key] = _metrics["plugin_requests"].get(key, 0) + 1


def get_metrics_snapshot():
    with _state_lock:
        total_requests = _metrics["total_requests"]
        avg_latency_ms = (
            _metrics["total_latency_ms"] / total_requests if total_requests else 0.0
        )
        return {
            "uptime_seconds": int(time.time() - _metrics["started_at"]),
            "total_requests": total_requests,
            "success_requests": _metrics["success_requests"],
            "failed_requests": _metrics["failed_requests"],
            "average_latency_ms": round(avg_latency_ms, 2),
            "plugin_requests": dict(_metrics["plugin_requests"]),
            "service_state": dict(_service_state),
        }


def resize_concurrency(max_concurrency):
    # 仅允许在初始化阶段调整；运行中动态调整需引入更复杂同步
    global request_semaphore
    safe_concurrency = max(1, int(max_concurrency))
    request_semaphore = threading.BoundedSemaphore(safe_concurrency)
    logger.info(f"api max concurrency set to {safe_concurrency}")
