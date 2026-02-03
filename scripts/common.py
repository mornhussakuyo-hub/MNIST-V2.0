import os

def validate_file_path(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not exists: {path}")
    return path

def safe_path_resolution(path):
    expanded=os.path.expanduser(path)
    absolute=os.path.abspath(expanded)
    normalized=os.path.normpath(absolute)
    return normalized