"""Minimal mmcv shim — only implements Config.fromfile for Python config files."""
import os, sys

class ConfigDict(dict):
    def keys(self):
        return super().keys()

class Config:
    def __init__(self, data: dict):
        self._data = ConfigDict(data)
    def keys(self):
        return self._data.keys()
    def __getitem__(self, key):
        return self._data[key]
    def __contains__(self, key):
        return key in self._data

    @staticmethod
    def fromfile(filepath: str) -> "Config":
        filepath = os.path.abspath(filepath)
        namespace = {}
        with open(filepath) as f:
            exec(compile(f.read(), filepath, "exec"), namespace)
        data = {k: v for k, v in namespace.items() if not k.startswith("_")}
        return Config(data)
