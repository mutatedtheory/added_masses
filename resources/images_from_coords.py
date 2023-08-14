import csv
import os

import numpy as np
from PIL import Image, ImageDraw

osp = os.path

# Path to the folder containing CSV files
current_dir = './'
out_dir = osp.join(current_dir, '..', 'images')

if not osp.isdir(out_dir):
    os.makedirs(out_dir)


def draw_image(coords, img_size, out_fn):
    """
    coords   : list of coordinates in 2D **pixels**
    img_size : width, height in pixels
    out_fn   : output file name
    """

    # Create a white background image
    image = Image.new("RGB", img_size, "white")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw the shape in black
    draw.polygon(coords, outline="black", fill="black")

    # Save the image
    image.save(out_fn)


def generate_image(curve_pts, png_fn, dx, dy, zoom_factor=0.5):
    """
    Generate Image using the dx, dy image size in pixels
    """

    nbpts = len(curve_pts)

    # Y should be inversed because imaging software scans from top
    # to bottom
    delta = np.array([dx*np.ones(nbpts), -dy*np.ones(nbpts)]).transpose()

    coords = curve_pts[:, :2] * delta * zoom_factor
    coords += [0.5*dx, 0.5*dy]
    coords = [tuple(vv) for vv in coords.astype(int).tolist()]

    draw_image(coords, (int(dx), int(dy)), png_fn)


# Iterate through CSV files in the folder
saved = []
fns = list(os.listdir(current_dir))
fns.sort()

for filename in fns:
    if 'csv' in filename[-4:]:
        csv_path = os.path.join(current_dir, filename)
        png_fn = osp.join(out_dir, os.path.splitext(filename)[0] + ".png")

        data = np.loadtxt(csv_path, delimiter=' ')
        saved.append((data, png_fn))

for data, png_fn in saved:
    print(png_fn)

    # Call custom function to generate image
    generate_image(data, png_fn, 1600, 1600)
