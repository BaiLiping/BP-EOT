"""
Extended Object Tracking with Elliptical Shapes - Compatibility Module

IMPORTANT: The main EOT algorithm implementation has been moved to the EO_Filter class.
This module now provides a compatibility wrapper for existing code.

For new code, use:
    from EO_Filter import EO_Filter
    eo_filter = EO_Filter(parameters)
    tracks, extents = eo_filter.track(measurements)
"""

from typing import List, Tuple
import numpy as np
from GenData import EOTParameters
from EO_Filter import EO_Filter


def eot_elliptical_shape(measurements_cell: List[np.ndarray], 
                        parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended Object Tracking with Elliptical Shapes (Compatibility Wrapper).
    
    This is a wrapper function that maintains compatibility with the original
    eot_elliptical_shape function call. The actual implementation is now in
    the EO_Filter class.
    
    Args:
        measurements_cell: List of measurement arrays, one per time step
        parameters: Algorithm parameters
        
    Returns:
        estimated_tracks: Estimated kinematic trajectories (4 x num_steps x num_tracks)
        estimated_extents: Estimated extent trajectories (2 x 2 x num_steps x num_tracks)
    """
    print("WARNING: Using compatibility wrapper. Consider updating to use EO_Filter class directly.")
    
    # Create EO_Filter instance and run tracking
    eo_filter = EO_Filter(parameters)
    return eo_filter.track(measurements_cell)