import win32gui
import win32con
import win32api
import time
import os

def wndproc(hwnd, msg, wparam, lparam):
    if msg == win32con.WM_DEVICECHANGE:
        if wparam == 0x8000:  # DBT_DEVICEARRIVAL
            with open("usb_trigger.txt", "w") as f:
                f.write(str(time.time()))
    return True

wc = win32gui.WNDCLASS()
wc.lpfnWndProc = wndproc
wc.lpszClassName = "USBDetector"
wc.hInstance = win32api.GetModuleHandle(None)
class_atom = win32gui.RegisterClass(wc)

hwnd = win32gui.CreateWindow(class_atom, "USBDetector", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)

# Initialize trigger file
with open("usb_trigger.txt", "w") as f:
    f.write("0")

# pump messages
win32gui.PumpMessages()
