__all__ = ['Event']


class Event:
    def __init__(self):
        self._listeners = set()

    @property
    def on(self):
        def decorator(func):
            self.connect(func)
            return func

        return decorator

    def connect(self, func):
        self._listeners.add(func)

    def disconnect(self, func):
        self._listeners.discard(func)

    def disconnect_all(self):
        self._listeners.clear()

    def emit(self, *args, **kwargs):
        for func in self._listeners:
            func(*args, **kwargs)
