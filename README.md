# Edge TPU in CPP
## Target environment
- Coral Dev Board
- Jetson Nano + USB Accelerator
- Raspberry Pi 3 + USB Accelerator


## How to build application code
```
# the following command is needed only once for a system
sudo apt install curl cmake

# the following commands are needed only once when you clone the code
cd EdgeTPU_CPP
git submodule init
git submodule update
cd third_party
chmod +x tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
# [This causes error ->] sh tensorflow/tensorflow/lite/tools/make/download_dependencies.sh

cd ../project_classification
mkdir build && cd build
cmake .. -DARCH_TYPE=armv7		## For Raspberry Pi
# cmake .. -DARCH_TYPE=aarch64	## For Jetson Nano, Coral Dev Board
make
mv libedgetpu.so.1.0 libedgetpu.so.1
sudo LD_LIBRARY_PATH=./ ./main

```

## How to create pre-built TensorflowLite library
## Environment
- Ubuntu (PC Linux)

## Commands
```
# Install tools
# sudo dpkg --add-architecture armhf
# sudo dpkg --add-architecture arm64
sudo apt update
sudo apt install -y build-essential crossbuild-essential-armhf crossbuild-essential-arm64
sudo apt install -y libusb-1.0-0-dev
# sudo apt install -y libusb-1.0-0-dev libusb-1.0-0-dev:arm64 libusb-1.0-0-dev:armhf

# Get Tensorflow code and build
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.0.0
cd ..
tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
tensorflow/tensorflow/lite/tools/make/build_generic_aarch64_lib.sh
tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh 

ls tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib
```

