#!/bin/bash
gnome-terminal --tab --title="chinese" -- bash -c "./chinese_classificator.py"
gnome-terminal --tab --title="french" -- bash -c "./french_classificator.py"
gnome-terminal --tab --title="mix" -- bash -c "./mix_classificator.py"