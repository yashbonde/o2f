"""utility functions for o2f
@yashbonde - 01.10.2020"""

import os
import time
import errno
import signal
import platform
import subprocess
from functools import wraps, partial

import torch
import numpy as np
import random
from maths import Math

def show_notification(title, text):
    """Send Desktop notifications on different platforms"""
    if platform.system() == "Darwin":
        subprocess.run(
            ["sh", "-c", f"osascript -e 'display notification \"{text}\" with title \"{title}\"'"])

    elif platform.system() == "Linux":
        subprocess.run(["notify-send", title, text])

    elif platform.system() == "Windows":
        try:
            from win10toast import ToastNotifier
        except ImportError as err:
            print('Error: to use Windows Desktop Notifications, you need to install `win10toast` first. Please run `pip install win10toast==0.9`.')

        toaster = ToastNotifier()
        toaster.show_toast(title, text, icon_path=None, duration=5)


# ---- util ---- #
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(
                signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, Math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
