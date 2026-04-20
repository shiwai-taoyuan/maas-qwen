from pydantic import BaseModel, Field


class MemoryResponse(BaseModel):
    ram: dict = Field(title="RAM", description="System memory stats")
    cuda: dict = Field(title="CUDA", description="nVidia CUDA memory stats")


class HeartBeatRequest(BaseModel):
    pass


class HeartBeatResponse(BaseModel):
    code: int = Field(default=0, title="code", description="接口状态码")
    msg: str = Field(default="alive", title="message", description="信息")


class MaaSBaseRequest(BaseModel):
    params: dict = Field(default_factory=lambda: {"plugin_id": 0}, title="Params", description="模型服务参数")
    task: str = Field(title="task", description="json str like {user_base64: xxx, template_base64: xxx,}")


class MaaSBaseResponse(BaseModel):
    code: int = Field(default=0, title="Model Id", description="接口状态码")
    status: int = Field(default=2, title="status id", description="任务状态,2是成功，3是失败")
    result: str = Field(default="返回结果", title="result", description="返回的结果json字符串")
    msg: str = Field(default="", title="message", description="异常信息")
