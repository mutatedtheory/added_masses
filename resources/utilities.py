# -----------------------------------------------------------------------------
# U T I L I T A R Y    F U N C T I O N S
# -----------------------------------------------------------------------------
import datetime
import inspect
import os
import pathlib
import time

from .prepro import load_dataset, load_datasets

DEBUG = False

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class BC:
    """
    Standard shell decoration characters
    """
    header = '\033[95m'
    okblue = '\033[94m'
    okcyan = '\033[96m'
    okgreen = '\033[92m'
    warning = '\033[93m'
    fail = '\033[91m'
    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    #
    warn = '\033[94m'
    error = '\033[91m'
    debug = '\033[96m'

def to_list(*args):
    """
    Return input value(s) as a list.
    """
    result = []
    for value in args:
        if isinstance(value, (list, tuple, dict)):
            result.extend(value)
        elif value is not None:
            result.append(value)
    return [i for i in result if i is not None]

def print_(mtype, message, tformat=None, **kwargs):
    """
    Print message to output with decoration and with adding a time stamp
    """
    # Handle list of messages
    if isinstance(message, (list, tuple)):
        for mm in message:
            print_(mtype, mm, tformat, **kwargs)
        return

    # Skip debug output ? (check global variable DEBUG)
    if mtype.lower() == 'debug':
        if not DEBUG:
            return

    # Terminal message with decoration and timestamp
    pf = getattr(BC, mtype, '')
    if tformat:
        pf = ''.join([getattr(BC, tf, '') for tf in to_list(tformat)])
    sf = BC.endc if pf else ''

    tstamp = datetime.datetime.now().strftime(" %H:%M:%S ")
    mlines = message.rstrip().split('\n')
    mlines = [message] if len(mlines) <= 1 else mlines # This allows blank lines to be printed
    for mline in mlines:
        message_full = f'[ {mtype.upper().ljust(5)} |{tstamp}] {pf}{mline}{sf}'
        print(message_full, **kwargs)

def print_nook(message):
    print_('error', f'[NOOK] {message}', ('bold', 'error'))

def print_ok(message):
    print_(' ', f'[ OK ] {message}', ('okcyan'))

def timing(f):
    """
    Decorator for profiling function run times
    """
    def wrap(*args, **kwargs):
        t1 = time.time()
        ret = f(*args, **kwargs)
        delta, unit = time.time() - t1, 's' # in seconds
        if delta <= 1.e-3:
            delta, unit = 1.e-3*delta, 'ms'

        stack = inspect.stack()[1]
        modstack = '.'.join(stack[1][1:].replace('.py','').split('/'))
        if 'mesher.' in modstack:
            modstack = modstack.split('mesher.')[1]

        print_('debug', '... {:.3f}{} in {}.[{}]\n'\
                        '              caller : {}.[{}]'.format(
            delta, unit, f.__module__, f.__name__,
            modstack, stack[3]))
        return ret
    return wrap


def list_files(root, ext):
    """
    Get a list of all files in the given directory (root) with
    the extension (ext)
    """
    return list(pathlib.Path(root).glob(f'*.{ext}'))


def new_dataset_id(start, length):
    """
    Geneartes a dataset ID based on current date and time
    """
    if DEBUG:
        ds_id = f"DS_DEBUG"
    else:
        ds_id = f"DS_{start+1}_{start+length}"
    ds_id = f"{ds_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print_('info', f'Automatically generated ID : {ds_id}')
    return ds_id
