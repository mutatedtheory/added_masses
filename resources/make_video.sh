#!/bin/bash
# Make a video from the png files, considers 1 frame for every 2 shapes
# and generates a video with a 30 fps rate...
ffmpeg -framerate 30 -i shape_%05d.png -vf "select='not(mod(n\,20))',setpts='N/FRAME_RATE/TB'" -c:v libx264 -r 30 output.mp4
