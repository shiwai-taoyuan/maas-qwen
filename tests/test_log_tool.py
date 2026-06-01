"""测试 tools/log_tool.py — Logger 初始化"""
import logging
import tempfile
from tools.log_tool import MyLogger, get_logger


class TestLoggerInit:
    def test_my_logger_name_is_maas_qwen(self):
        """验证 MyLogger 的 name 为 'maas-qwen'，而非 Logger 实例"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MyLogger(tmpdir, open_file_log=0, open_stream_log=1)
            assert logger.name == "maas-qwen"
            assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_singleton_with_correct_name(self):
        """验证 get_logger() 返回的 logger name 正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = get_logger(logger_dir=tmpdir, open_file_log=0, open_stream_log=1)
            # 第一次调用后重置，避免影响其他测试
            import tools.log_tool as lt
            lt.logger = None
            assert log.name == "maas-qwen"

    def test_logger_is_proper_logger_instance(self):
        """验证 MyLogger 是 logging.Logger 的有效子类"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MyLogger(tmpdir, open_file_log=0, open_stream_log=1)
            # 标准 logging 方法应可用
            logger.info("test message")
            logger.warning("test warning")
            logger.error("test error")
            assert True  # 没有异常即通过
