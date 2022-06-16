#!/bin/bash
gnome-terminal --tab --title="gaussian SVC chinese classificator" -- bash -c "./gauss_chin.py"
gnome-terminal --tab --title="gaussian SVC french classificator" -- bash -c "./gauss_fren.py"
gnome-terminal --tab --title="gaussian SVC mix classificator" -- bash -c "./gauss_mix.py"
gnome-terminal --tab --title="linear SVC french classificator" -- bash -c "./lin_fren.py"
gnome-terminal --tab --title="linear SVC chinese classificator" -- bash -c "./lin_chin.py"
gnome-terminal --tab --title="linear SVC mix classificator" -- bash -c "./lin_mix.py"