__all__ = ['format_dict']


class format_dict(dict):
    def __missing__(self, key):
        return '...'
