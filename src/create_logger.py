"""Create logger."""

import logging


def create_logger(logger_name,
                  include_stream_handler=True,
                  include_file_handler=True,
                  log_file='info.log',
                  log_level=logging.INFO,
                  log_format=None):
    """Create logger instance with given settings."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    handlers = []
    if include_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        handlers.append(stream_handler)
    if include_file_handler:
        file_handler = logging.FileHandler(filename=log_file, mode='w')
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    if log_format is not None:
        formatter = logging.Formatter(log_format)
        for handler in handlers:
            handler.setFormatter(formatter)

    for handler in handlers:
        logger.addHandler(handler)

    return logger
