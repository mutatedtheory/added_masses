# -----------------------------------------------------------------------------
# Generates a random shapes' dataset that is ready to be calculated
#
# Adapted from Viquerat's Shapes library on github
# (https://github.com/jviquerat/shapes)
#
# -----------------------------------------------------------------------------
# Credits/Copyright:
# - Hassan BERRO <hassan.berro@edf.fr>
# - Charbel HABCHI <habchi.charbel@gmail.com>
# -----------------------------------------------------------------------------
#
# Usage:
# ------
#
# python generate_dataset_parallel.py
#
# -----------------------------------------------------------------------------


import multiprocessing as mp
import os
import time
import traceback
from functools import partial

import numpy as np
import resources.utilities as ut

osp = os.path

#-------------------------------#
# U S E R   P A R A M E T E R S #
#-------------------------------#

ut.DEBUG     = True   # Activate/Deactivate Debug mode (messages and DS size)
DATASET_ID   = None   # If None, a dataset id is automatically generated

DATASET_SIZE = 10000  # Number of shapes to be generated in the dataset
DATASET_START = 30000 # Dataset starts then at (DATASET_START + 1)

NB_CPU = 1            # Number of processors to use for the dataset generation
#                     # Recommended : 4 processors if your system can handle

if ut.DEBUG:
    ut.print_('info', '------------------------------------------------------------')
    ut.print_('info', '    D E B U G     M O D E   I S    A C T I V A T E D', 'bold')
    ut.print_('info', '------------------------------------------------------------')
    ut.print_('info', 'In this mode the data set generated is limited to 100 shapes')
    ut.print_('info', 'Set ut.Debug = False to use your own parameters')
    ut.print_('info', '------------------------------------------------------------')
    DATASET_SIZE = min(100, DATASET_SIZE)
    DATASET_START = 0

# -----------------------------------------------------------------------------

BOX_DIMS     = [4., 4.]  # Image dimensions (DX, DY)
IMAGE_RES    = 400       # Image resolution in pixels per unit
SAMPLING_PTS = 40        # Number of Bezier sampling pts (higher = finer)

# -----------------------------------------------------------------------------
# Shape generation parameters, note that these input can be either static
# integers and floats or even functions. The script handles both

_int_3to7 = lambda: np.random.randint(3, 7)
_uniform = (lambda n: np.random.uniform(0., 1., size=n))

SHAPE_NPTS = 4           # Number of anchor points for Bezier generation
# SHAPE_NPTS = _int_3to7 # Example for setting a random variable for this param

SHAPE_RADIUS = _uniform  # Shape characteristic radius in [0, 1]
SHAPE_EDGY = 0.15        # Shape edgy parameter        in [0, 1]

SHAPE_SCALE  = 1.     # Corresponds to the Shapes library "magnify" parameter

# -----------------------------------------------------------------------------

def initialize():
    """
    Generate or verify the existance of the output directories
    """
    # Try loading the Shape class
    ut.print_('info', 'Loading the Shapes library')
    from resources.shapes import Shape
    ut.print_ok('Loaded')

    # Initialize output directories
    ut.print_('info', 'Initializing output directories')
    root = osp.join(osp.dirname(__file__), 'output')

    # Generate or use existing dataset ID
    dataset_id = ut.new_dataset_id(DATASET_START, DATASET_SIZE) \
                 if not DATASET_ID else DATASET_ID
    dataset_dir = osp.join(root, dataset_id)

    # Generate the output points directory name
    # (containing CSV information about each shape)
    points_dir = osp.join(dataset_dir, 'points')

    # Generate the output image directory name
    # (containing image snapshots for each shape)
    images_dir = osp.join(dataset_dir, 'images')

    # Generate output directories if they do not exist already
    for dirname in (root, dataset_dir, points_dir, images_dir):
        if not osp.isdir(dirname):
            os.makedirs(dirname)
            ut.print_('debug', f'Created       : {dirname}')
        else:
            ut.print_('debug', f'Already exists: {dirname}')

    ut.print_ok('Done')

    return points_dir, images_dir

def generate(start, end, points_dir, images_dir, verbose=True):
    """
    Actual generation script with start and end indices
    """
    xmin, xmax = -0.5*BOX_DIMS[0], 0.5*BOX_DIMS[0]
    ymin, ymax = -0.5*BOX_DIMS[1], 0.5*BOX_DIMS[1]
    image_pixels = BOX_DIMS[0]*IMAGE_RES, BOX_DIMS[1]*IMAGE_RES

    for idx in range(start, end):

        sid = f'shape_{idx+1:05d}'
        if verbose:
            ut.print_('info', sid)

        npts = SHAPE_NPTS if not callable(SHAPE_NPTS) \
               else SHAPE_NPTS()
        radius = [SHAPE_RADIUS]*npts if not callable(SHAPE_RADIUS) \
                 else SHAPE_RADIUS(npts)
        edgy = [SHAPE_EDGY]*npts if not callable(SHAPE_EDGY) \
                else SHAPE_EDGY(npts)

        shape = Shape(sid, npts, SAMPLING_PTS, radius, edgy)

        shape.generate(xmin=xmin, xmax=xmax,
                       ymin=ymin, ymax=ymax,
                       magnify=SHAPE_SCALE)

        csv_fn = osp.join(points_dir, f'{sid}.csv')
        np.savetxt(csv_fn, shape.curve_pts, delimiter=' ')
        if verbose:
            ut.print_ok('CSV') if osp.isfile(csv_fn) else ut.print_nook('CSV')

        png_fn = osp.join(images_dir, f'{sid}.png')

        shape.generate_image(png_fn, *image_pixels)

        if verbose:
            ut.print_ok('PNG') if osp.isfile(png_fn) else ut.print_nook('PNG')

class MPHandler:
    """
    Simple parallel processing handler that keeps a reference to running
    processes and is able to start them and tell if they are all done.
    """

    processes = None

    def __init__(self):
        """
        Initialize an empty list of processes
        """
        self.processes = []

    @property
    def number(self):
        """
        Returns the number of handled processes
        """
        return len(self.processes)

    def add(self, callback):
        """
        Append a new process to the hanlded list
        """
        process = mp.Process(None, callback, f'Thread [{self.number}]')
        self.processes.append(process)

    def start_all(self):
        """
        Start all processes at once
        """
        for process in self.processes:
            process.daemon = True
            process.start()

    def all_done(self):
        """
        Check if all referenced processes are finished while removing
        references to finished ones.
        """
        output = True
        for ith in range(self.number):
            if self.processes[ith] is None:
                continue

            if not self.processes[ith].is_alive():
                self.processes[ith] = None
            else:
                output = False

        return output


if __name__ == '__main__':
    """
    Main entry point when the script is executed in a python interpreter
    """

    ut.print_('info', 'Initializing', 'bold')
    try:
        dirpaths = initialize()
    except Exception as e:
        ut.print_nook('An error occurred')
        errors = ['-'*75]
        errors.extend(traceback.format_exc().strip().split('\n'))
        errors.append('-'*75)
        ut.print_('debug', errors)
        exit(4)

    ut.print_ok('All good')

    # Generate shapes
    import numpy as np
    from resources.shapes import Shape

    # Retrieve paths handled in initialization
    points_dir, images_dir = dirpaths

    nbcores = mp.cpu_count()
    if NB_CPU > nbcores:
        ut.print_('warn', f'The number of processes requested ({NB_CPU}) '\
                          f'exceeds the number of cores available ({nbcores}).')
        ut.print_('info', 'The dataset generation will proceed at your own risk...')
        ut.print_('info', 'Press Ctrl+C to Abort or wait 5 seconds', 'bold')
        time.sleep(5)

    # Generates shapes
    s = 's' if NB_CPU > 1 else ''
    ut.print_('info', f'Generating {DATASET_SIZE} shapes on {NB_CPU} proc{s}', 'bold')

    # Distribute the output distribution among the processors
    bunch = DATASET_SIZE//NB_CPU
    cuts = list(range(DATASET_START, DATASET_START+DATASET_SIZE, bunch))[:NB_CPU]
    cuts.append(DATASET_START+DATASET_SIZE)

    mp_handler = MPHandler()
    for icpu in range(NB_CPU):
        start, end = cuts[icpu:icpu+2]
        mp_handler.add(partial(generate, start, end, points_dir, images_dir, False))

    mp_handler.start_all()
    tref = time.time()

    # Wait for all threads are done before exiting
    sleep = min(4, 0.2 if ut.DEBUG else max(1, DATASET_SIZE/1000))
    while not mp_handler.all_done():
        done = len(os.listdir(images_dir))
        ratio = done/DATASET_SIZE
        elapsed = time.time()-tref
        remaining = f'{-elapsed + elapsed/ratio:.1f}s' \
                    if ratio > 0.005 else '-'.center(5)

        done_text = str(done).rjust(len(str(DATASET_SIZE)))
        perc_text = f'{100*ratio:04.1f}'.rjust(5)

        per_shape = f'{1000*elapsed/done:d}ms' \
                    if done > 0 else '-'.center(5)

        ut.print_(' ', f'[{perc_text}%] {done_text}/{DATASET_SIZE} generated'\
                 f' | {per_shape}/shape | ETA: {remaining}  ', end='\r')

        time.sleep(sleep)

    ut.print_(' ', '') # To avoid the last line being replaced...
    ut.print_ok(f'Generation is complete with {DATASET_SIZE} shapes')
    ut.print_('info', f'Total elapsed: {int(time.time()-tref)} s', 'bold')
