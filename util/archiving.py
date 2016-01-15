import os
import zipfile
from fnmatch import fnmatch


def zip_directory_structure(save_directories, zip_path, allowed_pattern=('*',), compression=zipfile.ZIP_DEFLATED):

    zf = zipfile.ZipFile(zip_path, "w", compression)
    if isinstance(allowed_pattern, str):
        allowed_pattern = (allowed_pattern, )
    if isinstance(save_directories, str):
        save_directories = [save_directories]
    for savedir in save_directories:
        for dirname, subdirs, files in os.walk(savedir):
            zf.write(dirname)
            for filename in files:
                if True in [fnmatch(filename, pat) for pat in allowed_pattern]:
                    zf.write(os.path.join(dirname, filename))
    zf.close()
    print('Saved zipfile containing %s to %s.' % (save_directories, zip_path))