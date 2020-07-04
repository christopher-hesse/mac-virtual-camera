# mac-virtual-camera

Simple virtual camera for mac

Uses a slightly modified version of https://github.com/johnboiles/coremediaio-dal-minimal-example

* `git clone https://github.com/christopher-hesse/mac-virtual-camera.git`
* `sudo rm -r /Library/CoreMediaIO/Plug-Ins/DAL/CMIOMinimalSample.plugin`
* `sudo cp -r built/CMIOMinimalSample.plugin /Library/CoreMediaIO/Plug-Ins/DAL/`
* restart chrome
* create a hangouts call
* hit the gear icon in the top right
* under the "video" selection select "CMIOMinimalSample Device"
* `pip install -e .`
* `python -m mac_virtual_camera.run`