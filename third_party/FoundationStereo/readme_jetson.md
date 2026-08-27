## Deployment on Jetson Devices
We would be using docker container from Isaac ROS and install different packages over it
* [Developer Environment Setup](https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html)
* Clone ```isaac_ros_common``` under ```${ISAAC_ROS_WS}/src```. 

    ```bash
    cd ${ISAAC_ROS_WS}/src && \
    git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git isaac_ros_common

    # download the image (running the container might fail; just ignore, we are using our own run script)
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```
* Run the container
    ```bash
    #!/bin/bash

    # Initialize X11 forwarding
    xhost +

    # Run the Docker container
    docker run \
    --privileged \
    --rm \
    -it \
    --network host \
    --ipc host \
    --gpus all \
    --runtime=nvidia \
    --name=foundation_stereo_container \
    -e DISPLAY \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e ISAAC_ROS_DEV_DIR=/workspace/isaac_ros-dev \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /tmp/.X11-unix:/tmp/.X11-unix:cached \
    -e DISPLAY="${DISPLAY}" \
    isaac_ros_dev-aarch64:latest
    ```
* Install in the container
    ```bash
    # check if this works or simply use the next command 
    pip install onnxruntime-gpu onnx

    # [!IMP] install onnxx runtime 
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

    pip3 install pycuda --user

    git clone https://github.com/onnx/onnx-tensorrt.git
    cd onnx-tensorrt
    python3 setup.py install

    pip install numpy==1.23.5
    
    export PYTHONPATH=$PYTHONPATH:<path_to_onnx-tensorrt>
    ```

* (optional for Jetson AGX Orin 64GB) Compile tensorrt
    ```bash
    # compiling models takes a lot of time (~2 hours with Jetson Orin 64GB)
    # Recommendation to use the compiled model foundation.engine for Jetson Orin 64GB instead of compiling
    trtexec --onnx=pretrained_models/foundation_stereo/foundation.onnx --verbose --saveEngine=pretrained_models/foundation_stereo/foundation.engine --fp16	
    ```

## Run inference
* Run from the root dir

    ```bash
    python scripts/run_demo_tensorrt.py \
            --left_img ${PWD}/assets/left.png \
            --right_img ${PWD}/assets/right.png \
            --save_path ${PWD}/output \
            --pretrained pretrained_models/foundation_stereo/foundation.engine \
            --hiera \
            --valid_iters 32 \
            --height 288 \
            --width 480 \
            --pc \
            --z_far 10.0
    ```
