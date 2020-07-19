# Play with tflite
Sample projects to use Tensorflow Lite for multi-platform

## Target Environment
- Platform
	- Linux (x64)
	- Linux (armv7)
	- Linux (aarch64)
	- Android (armv7)
	- Android (aarch64)
	- Windows (x64). Visual Studio 2017
	- (Only native build is supported)
- Delegate (todo)
	- Edge TPU
	- GPU

## How to build application
### Common (Get source code)
```sh
git clone https://github.com/iwatake2222/play_with_tflite.git
cd play_with_tflite

git submodule init
git submodule update
cd third_party/tensorflow
chmod +x tensorflow/lite/tools/make/download_dependencies.sh
tensorflow/lite/tools/make/download_dependencies.sh
```

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
	- `Where is the source code` : path-to-play_with_tflite/pj_tflite_standalone_cls_mobilenet_v2	(for example)
	- `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

**Note**
I found running with `Debug` causes exception, so use `Release` or `RelWithDebInfo` in Visual Studio.

### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```sh
cd pj_tflite_standalone_cls_mobilenet_v2   # for example
mkdir build && cd build
cmake ..
make
./main
```

## How to create pre-built TensorflowLite library
Pre-built TensorflowLite libraries are stored in `third_party/tensorflow_prebuilt` . You can use them, but if you want to build them by yourself, please use the following commands.

### Common (Get source code)
```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r2.3
# git checkout bb3c460114b13fda5c730fe43587b8e8c2243cd7  # This is a version I used to generate the libraries
```

### For Linux
You can create libtensorflow.so for x64, armv7 and aarch64. You can use the following commands in PC Linux like Ubuntu.

- Install tools (Bazel and Python)
	```sh
	sudo apt install bazel
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash ./Miniconda3-latest-Linux-x86_64.sh 
	conda create -n build_tflite
	conda activate build_tflite
	pip install python
	pip install numpy
	```
- Configuration
	```sh
	python configure.py 
	```
- Build
	- Build for x64 Linux
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so \
		-c opt \
		--copt -O3 \
		--strip always

		bazel build  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
		-c opt \
		--copt -DTFLITE_GPU_BINARY_RELEASE \
		--copt -DMESA_EGL_NO_X11_HEADERS \
		--copt -DEGL_NO_X11

		ls bazel-bin/tensorflow/lite/libtensorflowlite.so bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so -la
		```
	- Build for armv7 Linux (option is from build_pip_package_with_bazel.sh)
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so \
		-c opt \
		--config elinux_armhf \
		--copt -march=armv7-a \
		--copt -mfpu=neon-vfpv4 \
		--copt -O3 \
		--copt -fno-tree-pre \
		--copt -fpermissive \
		--define tensorflow_mkldnn_contraction_kernel=0 \
		--define raspberry_pi_with_neon=true \
		--strip always

		bazel build  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
		-c opt \
		--config elinux_armhf \
		--copt -DTFLITE_GPU_BINARY_RELEASE \
		--copt -DMESA_EGL_NO_X11_HEADERS \
		--copt -DEGL_NO_X11
		```
	- Build for aarch64 Linux
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so \
		-c opt \
		--config elinux_aarch64 \
		--define tensorflow_mkldnn_contraction_kernel=0 \
		--copt -O3 \
		--strip always

		bazel build  //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so \
		-c opt \
		--config elinux_aarch64 \
		--copt -DTFLITE_GPU_BINARY_RELEASE \
		--copt -DMESA_EGL_NO_X11_HEADERS \
		--copt -DEGL_NO_X11
		```

### For Windows
You can create libtensorflow.so(it's actually dll) for Windows. You can use the following commands in Windows 10 with Visual Studio. I used Visual Studio 2017 (You don't need to specify toolchain path. Bazel will automatically find it).

- Modify setting for bazel (workaround)
	- Reference: https://github.com/tensorflow/tensorflow/issues/28824#issuecomment-536669038
	- Edit `WORKSPACE` file just under the top layer of tensorflow
	```
	$ git diff
	diff --git a/WORKSPACE b/WORKSPACE
	index ea741c31c7..2115267603 100644
	--- a/WORKSPACE
	+++ b/WORKSPACE
	@@ -12,6 +12,13 @@ http_archive(
		],
	)

	+http_archive(
	+    name = "io_bazel_rules_docker",
	+    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
	+    strip_prefix = "rules_docker-0.7.0",
	+    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"],
	+)
	+
	# Load tf_repositories() before loading dependencies for other repository so
	# that dependencies like com_google_protobuf won't be overridden.
	load("//tensorflow:workspace.bzl", "tf_repositories")
	```

- Install tools
	- Install a environment providing linux like system (e.g. msys)
		- Add `C:\msys64\usr\bin` to Windows path
	- Install python environment (e.g. miniconda)
	- Install bazel (e.g. locate the bazel executable file into `C:\bin` )
		- Add `C:\bin` to Windows path
		- (I used bazel-3.4.1-windows-x86_64.zip)
- Configuration
	- Open python terminal (e.g. Anaconda Powershell Prompt)
		- (optional) create the new environment
		```sh
		conda create -n build_tflite
		conda activate build_tflite
		pip install python
		pip install numpy
		```
	- `cd path-to-tensorflow`
	```sh
	python configure.py 
	```
- Build
	- Build for Windows
		```sh
		bazel build //tensorflow/lite:libtensorflowlite.so `
		-c opt `
		--copt -O3 `
		--strip always 

		# bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so `
		# -c opt `
		# --copt -O3 `
		# --copt -DTFLITE_GPU_BINARY_RELEASE `
		# --strip always 
		```

### For Android
You can create libtensorflow.so for Android (armv7, aarch64) both in PC Linux and in Windows. I used Windows 10 to build.
You need to install Android SDK and Android NDK beforehand, then specify the path to sdk and ndk when `python configure.py` asks "`Would you like to interactively configure ./WORKSPACE for Android builds`". (Notice that path must be like `c:/...` instead of `c:\...`)

- Build for armv7 Android
	```sh
	python configure.py 
	bazel build //tensorflow/lite:libtensorflowlite.so `
	-c opt `
	--config android_arm `
	--copt -O3 `
	--strip always 

	bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so `
	-c opt `
	--config android_arm `
	--copt -O3 `
	--copt -DTFLITE_GPU_BINARY_RELEASE `
	--strip always 
	```
- Build for aarch64 Android
	```sh
	python configure.py 
	bazel build //tensorflow/lite:libtensorflowlite.so `
	-c opt `
	--config android_arm64 `
	--copt -O3 `
	--strip always 

	bazel build //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so `
	-c opt `
	--config android_arm64 `
	--copt -O3 `
	--copt -DTFLITE_GPU_BINARY_RELEASE `
	--strip always 
	```

## Acknowledgement
- References:
	- https://www.tensorflow.org/lite/performance/gpu_advanced
	- https://www.tensorflow.org/lite/guide/android
	- https://qiita.com/terryky/items/fa18bd10cfead076b39f
	- https://github.com/terryky/tflite_gles_app

- This project includes output files (such as `libtensorflowlite.so`) of the following project:
	- https://github.com/tensorflow/tensorflow
- This project includes models:
	- mobilenetv2-1.0
		- https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
		- https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
		- http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
		- https://coral.withgoogle.com/models/
		- https://www.tensorflow.org/lite/guide/hosted_models
	- coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
		- https://www.tensorflow.org/lite/models/object_detection/overview
		- https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
	- deeplabv3_mnv2_dm05_pascal_quant
		- https://github.com/google-coral/edgetpu/tree/master/test_data
		- https://github.com/google-coral/edgetpu/blob/master/test_data/deeplabv3_mnv2_dm05_pascal_quant.tflite
		- https://github.com/google-coral/edgetpu/blob/master/test_data/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite
