#!/bin/bash

XSOCK=/tmp/.X11-unix
XAUTH=/root/.Xauthority

DOCKER_REPO=kakalin/
BRAND=tha
VERSION=devel
IMAGE_NAME=${DOCKER_REPO}${BRAND}:${VERSION}

# https://stackoverflow.com/questions/3162385/how-to-split-a-string-in-shell-and-get-the-last-field
PROJECT_FOLDER=${PWD##*/}
ENTRYPOINT=./entrypoint.sh

GPU_DEVICE=""
if [[ -z $1 ]]; then
    GPU_DEVICE="all"
else
    GPU_DEVICE="device=$1"
fi

docker_run_params=$(cat <<-END
    --rm -it \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v $XSOCK:$XSOCK \
    -v $HOME/.Xauthority:$XAUTH \
    --network=host \
    -p 8888:8888 \
    -p 6006:6006 \
    -v $PWD:/root/$PROJECT_FOLDER \
    -w /root/$PROJECT_FOLDER \
    --entrypoint $ENTRYPOINT \
    $IMAGE_NAME
END
)

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    GPU=$(lspci | grep -i '.* vga .* nvidia .*')
    if [[ $GPU == *' NVIDIA '* ]]; then
        # printf 'Nvidia GPU is present:  %s\n' "$GPU"
        docker run \
            --gpus $GPU_DEVICE \
            -e DISPLAY=$DISPLAY \
            -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
            -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
            $docker_run_params
    else
        docker run \
            -e QT_QUICK_BACKEND=software \
            -e DISPLAY=$DISPLAY \
            -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
            -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
            $docker_run_params
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    docker run \
        -e DISPLAY=host.docker.internal:0 \
        -e QT_QUICK_BACKEND=software \
        $docker_run_params
else
    echo "Haven't supported this OS Type, please check command\n and update it."
fi
