#!/bin/bash

pushd `pwd` > /dev/null
cd `dirname $0`
echo "Working Path: "`pwd`

ROS_VERSION="ROS2"
ROS_HUMBLE="humble"

echo "ROS version is: "$ROS_VERSION

# clear `build/` folder.
# TODO: Do not clear these folders, if the last build is based on the same ROS version.
rm -rf ../../build/
rm -rf ../../devel/
rm -rf ../../install/
# clear src/CMakeLists.txt if it exists.
if [ -f ../CMakeLists.txt ]; then
    rm -f ../CMakeLists.txt
fi

# exit

# build
pushd `pwd` > /dev/null

cd ../../
#colcon build --cmake-args -DROS_EDITION=${ROS_VERSION} -DHUMBLE_ROS=${ROS_HUMBLE}

colcon build --cmake-args -DHUMBLE_ROS=${ROS_HUMBLE}

popd > /dev/null
