import subprocess
import sys

IS_WIN32 = 'win32' in str(sys.platform).lower()


def subprocess_call(*args, **kwargs):
    # also works for Popen. It creates a new *hidden* window, so it will work in
    # frozen apps (.exe).
    if IS_WIN32:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = (
            subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        )
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
    return_code = subprocess.call(*args, **kwargs)
    print(return_code)
    return return_code
