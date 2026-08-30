
"""Logging setup for the long-running pipeline stages.

A forward run takes about an hour and prints one timed line per snapshot, so
the entry points configure a single logger here rather than using bare
``print``. Timestamps make it possible to attribute wall-clock cost to
individual snapshots after the fact.

Note this module shadows the standard library's ``logging`` inside the package
namespace; it is always imported as
``baqaro.utils.logging``, and its own ``import logging`` picks up
the standard library as usual.
"""

import logging


def set_logger(path=None, print_to_console=True):
    """Configure and return the pipeline logger.

    Safe to call more than once: existing handlers are cleared first, so
    repeated calls reconfigure rather than accumulate.

    Parameters
    ----------
    path : str or None
        If given, also write to this file. If None, console only.
    print_to_console : bool
        Attach a stream handler to stderr.

    Returns
    -------
    logging.Logger
        Logger emitting ``"<timestamp> - <message>"`` at INFO level.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # getLogger(__name__) returns the SAME logger object on every call, so
    # without this a second set_logger() call would stack a second set of
    # handlers → every line printed twice (thrice, …). Clear first so each
    # call fully (re)configures the handlers. Also disable propagation to the
    # root logger to avoid a duplicate emit there.
    for _h in list(logger.handlers):
        logger.removeHandler(_h)
        try:
            _h.close()
        except Exception:
            pass
    logger.propagate = False

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if print_to_console:
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if path is None:
        return logger
    else:
        # Create a file handler
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)

        return logger

