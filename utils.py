import platform
import os
import subprocess

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
