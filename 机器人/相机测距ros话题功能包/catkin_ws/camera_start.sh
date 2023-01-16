#!/bin/bash

sleep 5s
source /opt/ros/noetic/setup.bash
gnome-terminal -t "camera_start.sh" -x bash -c "roscore;exec bash;"
sleep 5s
source /userdata/work/catkin_ws/devel/setup.bash
gnome-terminal -t "camera_start.sh" -x bash -c "rosrun camera_distance detect.py;exec bash;"
