"""This is a custom exception that can be used for Startup issues.
"""


class StartupError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
