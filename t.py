from pyv4l2.control import Control, ControlList

controls = ControlList("/dev/video0")

for c in controls:
    print(c)
