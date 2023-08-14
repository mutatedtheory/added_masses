# -----------------------------------------------------------------------------
# Utilitary function (load_dataset) for loading dataset results
# - images in images/*.png are loaded and resized as a single channel
#   numpy array
# - added masses in calcs/*.csv files are read into 3 vectors mx, my, mxy
# -----------------------------------------------------------------------------
# Credits/Copyright:
# - Hassan BERRO <hassan.berro@edf.fr>
# - Charbel HABCHI <habchi.charbel@gmail.com>
# -----------------------------------------------------------------------------
#
# Usage:
# ------
#
# from resources.prepro import load_datasets, load_dataset
#
# images = load_dataset(DATASET_ID, only_images=True)
# images, values = load_dataset(DATASET_ID, only_images=False)
#
# Alternatively (recommended), you can access the function as follows:
# from resources import utilities as ut
# images = ut.load_dataset(DATASET_ID, only_images=True)
#          \--\
#            \_ the load_dataset function is imported in utilities
#
# Multiple datasets can be loaded at once
# images, values = ut.load_datasets(['name1', 'name2'])
#
# -----------------------------------------------------------------------------

import os

import numpy as np
from PIL import Image

from . import utilities as ut

osp = os.path


def load_datasets(dataset_ids, img_size=(299, 299)):
    """
    Loads multiple datasets and concatenates their data points
    """
    images, values = [], []
    ut.print_('info', f'Requested to load {len(dataset_ids)} datasets...', 'bold')
    for dataset_id in dataset_ids:
        ds_images, ds_values = load_dataset(dataset_id, img_size, False)
        images.append(ds_images)
        values.append(ds_values)

    images = np.rowstack(images)
    values = np.rowstack(values)

    ut.print_ok('All datasets were loaded and concatenated')
    return images, values


def load_dataset(dataset_id, img_size=(299, 299), only_images=False):
    """
    Preprocessing function for loading a dataset (images and optionnally calcs)
    """
    # Check if a preload has been done on this dataset
    ut.print_('info', f'Handling {dataset_id}', 'bold')
    ut.print_(' ', f'Checking for previous loads')
    img_save_fn = osp.join('output', dataset_id, 'images.npz')
    val_save_fn = osp.join('output', dataset_id, 'values.npz')

    images, values = None, None
    if osp.isfile(img_save_fn):
        images = np.load(img_save_fn)['images']

    if osp.isfile(val_save_fn):
        values = np.load(val_save_fn)['values']

    if only_images:
        if images is not None:
            ut.print_ok('Loaded .npz for images from previous save')
            return images
    else:
        if images is not None and values is not None:
            ut.print_ok('Loaded .npz for images and masses from previous save')
            return images, values

    ut.print_(' ', '.npz files were not found, loading images and csvs')

    # Actual loading
    images_dir = osp.join('output', dataset_id, 'images')
    calcs_dir = osp.join('output', dataset_id, 'calcs')

    ref_dir = images_dir if only_images else calcs_dir
    ref_ext = '.png' if only_images else '.csv'

    fns = list(os.listdir(ref_dir))
    fns.sort()

    # Read dataset (images and optionnaly csv files of results too)
    images = np.zeros((len(fns), img_size[0], img_size[1], 1))

    if not only_images:
        values = np.zeros((len(fns), 3)) # MX; MY; MXY

    for i, fn in enumerate(fns):
        if fn.endswith(ref_ext):
            name_no_ext = os.path.splitext(fn)[0]
            png_fn = osp.join(images_dir, f'{name_no_ext}.png')

            img = Image.open(png_fn)
            img = img.resize((img_size[0], img_size[1]))          # Resize the image to desired dimensions
            images[i, :, :, 0] = np.array(img)[:, :, 0] / 255.0            # Normalize pixel values to [0, 1]

            if not only_images:
                csv_fn = os.path.join(calcs_dir, f'{name_no_ext}.csv')
                masses = np.loadtxt(csv_fn, delimiter=' ')
                values[i, :] = (masses[0, 0], masses[1, 1], masses[0, 1])

    ut.print_ok(f'Loaded {dataset_id}...')
    ut.print_(' ', 'Saving .npz files for future needs')

    np.savez(img_save_fn, images=images)

    if only_images:
        return images

    np.savez(val_save_fn, values=values)

    return images, values
