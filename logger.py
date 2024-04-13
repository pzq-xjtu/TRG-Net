import logging
import os

class Logger():
    def __init__(self, path,level="DEBUG"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.path = path

    def console_handler(self,level="DEBUG"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        console_handler.setFormatter(self.get_formatter()[0])

        return console_handler

    def file_handler(self, level="DEBUG"):
        file_handler = logging.FileHandler(os.path.join(self.path, "log.log"),mode="a",encoding="utf-8")
        file_handler.setLevel(level)

        file_handler.setFormatter(self.get_formatter()[1])

        return file_handler

    def get_formatter(self):

        console_fmt = logging.Formatter(fmt="%(asctime)s--->%(message)s")
        file_fmt = logging.Formatter(fmt="%(asctime)s--->%(message)s")

        return console_fmt,file_fmt

    def get_log(self):
        self.logger.addHandler(self.console_handler())
        self.logger.addHandler(self.file_handler())

        return self.logger


