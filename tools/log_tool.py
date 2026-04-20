import logging
import os
import sys

from concurrent_log_handler import ConcurrentRotatingFileHandler

"""
-------------------------------------------------
Description: 日志工具
-------------------------------------------------
"""


class MyLogger(logging.Logger):
    """
    %(name)s            Logger的名字
    %(levelno)s         数字形式的日志级别
    %(levelname)s       文本形式的日志级别
    %(pathname)s        调用日志输出函数的模块的完整路径名，可能没有
    %(filename)s        调用日志输出函数的模块的文件名
    %(module)s          调用日志输出函数的模块名
    %(funcName)s        调用日志输出函数的函数名
    %(lineno)d          调用日志输出函数的语句所在的代码行
    %(created)f         当前时间，用UNIX标准的表示时间的浮 点数表示
    %(relativeCreated)d 输出日志信息时的，自Logger创建以 来的毫秒数
    %(asctime)s         字符串形式的当前时间。默认格式是 “2026-04-20 16:49:45,896”。逗号后面的是毫秒
    %(thread)d          线程ID。可能没有
    %(threadName)s      线程名。可能没有
    %(process)d         进程ID。可能没有
    %(message)s         用户输出的消息
    """

    def __init__(self, file_dir, open_file_log=1, file_log_level="DEBUG",
                 open_stream_log=1, stream_log_level="DEBUG",
                 simple_mode=True):
        super().__init__(self)
        self.level_list = ["DEBUG", "INFO", "WARN", "ERROR"]
        self.sep = "-"
        self.max_bytes = 1024 * 1024 * 30
        self.max_count = 10
        self.simple_mode = simple_mode
        self.open_file_log = open_file_log
        self.open_stream_log = open_stream_log
        self.file_log_level = file_log_level if file_log_level in self.level_list else "DEBUG"
        self.stream_log_level = stream_log_level if stream_log_level in self.level_list else "DEBUG"
        os.makedirs(file_dir, exist_ok=True)
        self.file_dir = file_dir
        format_str = self.get_simple_format() if simple_mode else self.get_detail_format()
        formatter = logging.Formatter(format_str)
        if self.open_file_log:
            debug_handler = ConcurrentRotatingFileHandler(os.path.join(self.file_dir, "run.log"),
                                                          maxBytes=self.max_bytes, backupCount=self.max_count,
                                                          encoding="utf8")
            debug_handler.setLevel(self.file_log_level)
            debug_handler.setFormatter(formatter)
            self.addHandler(debug_handler)

            error_handler = ConcurrentRotatingFileHandler(os.path.join(self.file_dir, "error.log"),
                                                          maxBytes=self.max_bytes, backupCount=self.max_count,
                                                          encoding="utf8")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.addHandler(error_handler)
        if self.open_stream_log:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self.stream_log_level)
            ch.setFormatter(formatter)
            self.addHandler(ch)

    def get_simple_format(self):
        format_list = ["[%(asctime)s]", "[%(filename)s:%(lineno)d]", "[%(levelname)s]", "%(message)s"]
        return self.sep.join(format_list)

    def get_detail_format(self):
        format_list = ["[%(asctime)s]", "[%(filename)s:%(lineno)d]", "[process:%(process)d thread:%(thread)d]",
                       "[%(levelname)s]", "%(message)s"]
        return self.sep.join(format_list)


logger = None


def get_logger(logger_dir="logs", open_file_log=1, file_log_level="DEBUG", open_stream_log=1, stream_log_level="DEBUG",
               *args, **kwargs):
    global logger
    if not logger:
        logger = MyLogger(logger_dir, open_file_log=open_file_log, file_log_level=file_log_level,
                          open_stream_log=open_stream_log, stream_log_level=stream_log_level, *args, **kwargs)
    return logger
