from collections import defaultdict
from typing import Dict, Sequence

import numpy as np
from numpy.typing import NDArray
from pyvbmc import VBMC

def get_timings_pytree(vbmc: VBMC) -> Dict[str, NDArray[np.float_]]:
    """
    Return VBMC timing information grouped by timed operation.

    This representation loses knowledge of during which iteration a timing was measured,
    but is more amenable for aggregating.
    """
    aggregate = defaultdict(list)
    # Collect timing information from each iteration to a dict of lists
    for timer in vbmc.iteration_history['timer']:
        for k, v in timer._durations.items():
            aggregate[k].append(v)

    # Convert to pytree (dict of ndarrays)
    return dict((k, np.asarray(v)) for k, v in aggregate.items())

def _get_evaluated_points(vbmc: VBMC):
    mask = np.all(~np.isnan(vbmc.function_logger.X_orig), axis=-1)
    X = vbmc.function_logger.X_orig[mask]
    y = vbmc.function_logger.y_orig[mask]
    return X, y

def count_failed_evaluations(vbmc: VBMC) -> int:
    """
    Return the number of failed (i.e. nonfinite) target evaluations observed by VBMC.
    """
    _, y = _get_evaluated_points(vbmc)
    return np.count_nonzero(~np.isfinite(y))

__all__ = ["get_timings_pytree", "count_failed_evaluations"]
