import logging
import json


class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, level, message, **kwargs):
        log_entry = {
            "timestamp": logging.Formatter().formatTime(logging.LogRecord(
                name=self.logger.name, level=level, pathname="", lineno=0, msg="", args=(), exc_info=None)),
            "level": logging.getLevelName(level),
            "message": message,
            **kwargs
        }
        self.logger.log(level, json.dumps(log_entry,ensure_ascii=False))


# 使用示例
# logger = StructuredLogger(__name__)
# logger.log(logging.INFO, "用户操作", user_id=123, action="login", status="success")
