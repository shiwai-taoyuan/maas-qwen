#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:

from fastapi import APIRouter, FastAPI, Request, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from .models import *
import json
from configs import (
    logger,
    FTP_DIR,
    TMP_OUTPUT_DIR,
    API_ACQUIRE_TIMEOUT_SECONDS,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_UPLOAD_EXTENSIONS,
    ENABLE_DOCS,
)
from typing import Any, Dict
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.staticfiles import StaticFiles
import configs
from pathlib import Path
import uvicorn
from starlette.responses import FileResponse
from maas_model_source import model_function_register
from server import runtime_state
import time
from uuid import uuid4


def api_middleware(app: FastAPI):
    rich_available = True
    try:
        import anyio  # importing just so it can be placed on silent list
        import starlette  # importing just so it can be placed on silent list
        from rich.console import Console
        console = Console()
    except:
        import traceback
        rich_available = False

    def handle_exception(request: Request, e: Exception):
        error_id = str(uuid4())
        err = {
            "error_id": error_id,
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        logger.error(f"API error: {request.method} {request.url} {err}")
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            if rich_available:
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[anyio, starlette],
                                        word_wrap=False, width=min([console.width, 200]))
            else:
                traceback.print_exc()
        rsp = MaaSBaseResponse(code=-1, msg=str(jsonable_encoder(err)), status=3)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=json.loads(rsp.json()))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI):
        self.router = APIRouter()
        self.app = app
        static_dir = Path(configs.PROJECT_DIR) / "static"
        self.use_local_docs_assets = static_dir.exists()
        if self.use_local_docs_assets:
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        api_middleware(self.app)
        self.add_api_route("/heartBeat", self.heart_beat_api, methods=["GET", "POST"],
                           response_model=HeartBeatResponse)
        self.add_api_route("/healthz", self.healthz, methods=["GET"])
        self.add_api_route("/readyz", self.readyz, methods=["GET"])
        self.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.add_api_route("/anything", self.anything, methods=["GET", "POST"])
        self.add_api_route("/queryAndBody", self.query_and_body, methods=["GET", "POST"])
        if ENABLE_DOCS:
            self.add_api_route("/docs", self.custom_swagger_ui_html, methods=["GET", "POST"])
        self.add_api_route("/ftp/{file_name:path}", self.download, methods=["GET"])
        self.add_api_route("/upload", self.upload, methods=["POST"])
        self.add_api_route("/modelFunction", self.template_function, methods=["POST"],
                           response_model=MaaSBaseResponse)
        self.add_api_route("/v1/qwen2Completion", self.qwen2_chat_sync, methods=["POST"],
                           response_model=MaaSBaseResponse)

    def qwen2_chat_sync(self, req: MaaSBaseRequest):
        status = 2
        completion = ""
        msg = ""
        code = 0
        plugin_id = 0
        success = False
        acquired = runtime_state.request_semaphore.acquire(timeout=API_ACQUIRE_TIMEOUT_SECONDS)
        if not acquired:
            raise HTTPException(status_code=429, detail="server is busy, please retry later")
        start_time = time.perf_counter()
        try:
            logger.info(req)
            params = req.params or {"plugin_id": 0}
            plugin_id = int(params.get("plugin_id", 0))
            do_sample = params.get("do_sample", True)
            max_length = params.get("max_length", 8192)
            top_p = params.get("top_p", 0.8)
            temperature = params.get("temperature", 0.95)
            model_function = model_function_register.get(plugin_id)
            if model_function is None:
                raise HTTPException(status_code=400, detail=f"plugin_id {plugin_id} not found")
            task = json.loads(req.task)
            history = task.get("history", [])
            completion = model_function(
                query=task.get("context", ""),
                history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
            )
            logger.info(completion)
            success = True
        except HTTPException:
            raise
        except Exception as err:
            logger.exception("")
            status = 3
            msg = str(jsonable_encoder(err))
            code = -1
        finally:
            runtime_state.request_semaphore.release()
            latency_ms = (time.perf_counter() - start_time) * 1000
            runtime_state.record_request(latency_ms=latency_ms, success=success, plugin_id=plugin_id)
        return MaaSBaseResponse(code=code, result=completion, status=status, msg=msg)

    def custom_swagger_ui_html(self, req: Dict[Any, Any]):
        if not self.use_local_docs_assets:
            return get_swagger_ui_html(
                openapi_url=self.app.openapi_url,
                title="Swagger UI",
                oauth2_redirect_url=self.app.swagger_ui_oauth2_redirect_url,
            )
        return get_swagger_ui_html(
            openapi_url=self.app.openapi_url,
            title="Swagger UI",
            oauth2_redirect_url=self.app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
            swagger_favicon_url="/static/favicon.png",
        )

    def download(self, file_name):
        """
        直接通过 HTTP GET 下载 FTP_DIR 下面的文件
        :param file_name:
        :return:
        """
        base_dir = Path(FTP_DIR).resolve()
        file_path = (base_dir / file_name).resolve()
        if base_dir not in file_path.parents and file_path != base_dir:
            raise HTTPException(status_code=400, detail="invalid file path")
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        logger.info(f"request to download file = {file_path}")
        fr = FileResponse(path=str(file_path), filename=file_path.name)
        return fr

    def upload(self, file: UploadFile = File(...)):
        safe_filename = Path(file.filename).name
        suffix = Path(safe_filename).suffix.lower()
        if not safe_filename:
            raise HTTPException(status_code=400, detail="empty filename")
        if suffix and suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"file extension {suffix} is not allowed")

        total_size = 0
        destination = Path(TMP_OUTPUT_DIR) / safe_filename
        try:
            with destination.open("wb") as buffer:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_SIZE_BYTES:
                        buffer.close()
                        destination.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="file too large")
                    buffer.write(chunk)
            return JSONResponse(content={"message": "文件上传成功", "file_name": safe_filename}, status_code=200)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("upload failed")
            raise HTTPException(status_code=500, detail=f"upload failed: {str(e)}")
        finally:
            file.file.close()

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def heart_beat_api(self, req: HeartBeatRequest = None):
        return HeartBeatResponse()

    def healthz(self):
        return {"status": "alive"}

    def readyz(self):
        state = runtime_state.get_service_state()
        if not state.get("ready", False):
            raise HTTPException(status_code=503, detail="service not ready")
        return {"status": "ready", "state": state}

    def metrics(self):
        return runtime_state.get_metrics_snapshot()

    def anything(self, req: Dict[Any, Any]):
        """
        接收任意请求，并返回
        :param req:
        :return:
        """
        logger.info(req)
        return req

    def query_and_body(self, id: int, req: MaaSBaseRequest):
        """
        接收url和body参数，并返回
        :param id:
        :param req:
        :return:
        """
        logger.info(f"id:= {id}")
        logger.info(f"req:= {req}")
        return (id, req)

    def template_function(self, req: MaaSBaseRequest = None):
        logger.info(f"receive request {str(req)}")
        task = json.loads(req.task)

        # write the real function here

        result = dict()
        rsp = MaaSBaseResponse()
        rsp.result = json.dumps(result, ensure_ascii=False)
        return rsp

    def launch(self, server_name, port, log_level="info"):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port, log_level=log_level)
