from warnings import warn

from bio_rtd import logger


class TestLogger(logger.RtdLogger):
    """ implementation of RtdLogger for testing """

    def __init__(self,
                 log_level: int = logger.RtdLogger.DEBUG,
                 log_data: bool = True):
        super().__init__(log_level, log_data)

    def log(self, level: int, msg: str):
        if level == logger.RtdLogger.WARNING:
            warn(Warning(msg))
        elif level == logger.RtdLogger.INFO:
            warn(Warning("Info: " + msg))
        elif level == logger.RtdLogger.ERROR:
            raise RuntimeError(msg)

    def _on_data_stored(self, level: int, tree: dict, key: str, value: any):
        pass


class EmptyLogger(logger.RtdLogger):
    """ empty RtdLogger for testing (no returns) """

    def log(self, level: int, msg: str):
        pass

    def _on_data_stored(self, level: int, tree: dict,
                        key: str, value: any):  # pragma: no cover
        pass
