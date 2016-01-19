import os


def list_dirs(root, fullpath=False, exceptions=None):
    if exceptions is None:
        exceptions = []
    elif isinstance(exceptions, str):
        exceptions = [exceptions]

    if fullpath:
        dirs = [os.path.join(root, name) for name in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, name))
                and name not in exceptions]
    else:
        dirs = [name for name in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, name))
                and name not in exceptions]

    return dirs
