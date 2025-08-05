# Extended Object Tracking - Python Implementation

This folder contains Python implementations of the Extended Object Tracking (EOT) algorithms originally implemented in MATLAB by Florian Meyer (2020).

## Reference Paper
F. Meyer and J. L. Williams, "Scalable detection and tracking of geometric extended objects," IEEE Trans. Signal Process., vol. 69, pp. 6283–6298, Oct. 2021.

## Python Classes

### 1. DataGenerator Class (`DataGenerator.py`)

Handles simulation of extended object scenarios including:
- **Target trajectory generation** with constant velocity motion model
- **Extent simulation** using inverse Wishart distributions  
- **Measurement generation** with Poisson-distributed detections and clutter
- **Complete scenario simulation** pipeline

**Key Methods:**
- `get_start_states()`: Generate initial target states and extent matrices
- `generate_tracks_unknown()`: Simulate target trajectories with appearance/disappearance
- `generate_cluttered_measurements()`: Generate noisy measurements with false alarms
- `simulate_scenario()`: Complete simulation pipeline

### 2. ExtendedObjectFilter Class (`ExtendedObjectFilter.py`)

Implements the belief propagation-based extended object tracking filter:
- **Particle filtering** for kinematic state estimation
- **Inverse Wishart filtering** for extent estimation
- **Belief propagation** for data association
- **Track management** with birth, death, and pruning

**Key Methods:**
- `eot_elliptical_shape()`: Main filtering algorithm
- `perform_prediction()`: Predict particles to next time step
- `update_particles()`: Update particles with measurements
- `track_formation()`: Form continuous tracks from estimates
- `run_filter()`: Complete filtering pipeline

### 3. Utility Functions (`utils.py`)

Helper functions for visualization and evaluation:
- `show_results()`: Visualize tracking results
- `error_ellipse()`: Generate uncertainty ellipses
- `compute_ospa_distance()`: OSPA distance metric
- `generate_default_parameters()`: Default parameter setup

## Usage Examples

### Basic Usage

```python
import numpy as np
from DataGenerator import DataGenerator
from ExtendedObjectFilter import ExtendedObjectFilter
from utils import generate_default_parameters, set_prior_extent

# Setup parameters
parameters = generate_default_parameters()
parameters = set_prior_extent(parameters, mean_target_dimension=3.0)

# Generate simulation data
data_gen = DataGenerator(parameters)
target_tracks, target_extents, measurements = data_gen.simulate_scenario(
    num_steps=50, num_targets=3, mean_target_dimension=3.0,
    start_radius=75, start_velocity=10, 
    appearance_from_to=np.array([[1, 50], [10, 45], [5, 40]]).T
)

# Run tracking filter
eot_filter = ExtendedObjectFilter(parameters)
estimated_tracks, estimated_extents = eot_filter.run_filter(measurements)
```

### Parameter Configuration

The parameter dictionary includes:

**Simulation Parameters:**
- `scanTime`: Time between scans (default: 0.2s)
- `accelerationDeviation`: Process noise level (default: 1.0)
- `surveillanceRegion`: Tracking area bounds
- `measurementVariance`: Sensor noise variance
- `meanMeasurements`: Expected detections per target
- `meanClutter`: Expected false alarms per scan

**Filter Parameters:**
- `numParticles`: Number of particles (default: 5000)
- `detectionThreshold`: Existence probability threshold
- `thresholdPruning`: Pruning threshold
- `numOuterIterations`: Belief propagation iterations

**Prior Parameters:**
- `priorExtent1`, `priorExtent2`: Inverse Wishart prior parameters
- `priorVelocityCovariance`: Velocity prior covariance

## Test Scripts

### `test_python_classes.py`
Comprehensive test replicating the MATLAB `main.m` functionality:
```bash
python test_python_classes.py
```

### `example_usage.py`  
Simple example with reduced complexity:
```bash
python example_usage.py
```

## Dependencies

Required Python packages:
- `numpy`: Numerical computations
- `scipy`: Statistical distributions and optimization
- `matplotlib`: Visualization

Install with:
```bash
pip install numpy scipy matplotlib
```

## Key Differences from MATLAB Implementation

1. **Array Indexing**: Python uses 0-based indexing vs MATLAB's 1-based
2. **Matrix Operations**: Uses numpy for linear algebra operations
3. **Data Structures**: Uses dictionaries for parameters vs MATLAB structs
4. **Visualization**: Uses matplotlib instead of MATLAB plotting
5. **Random Sampling**: Uses scipy.stats for statistical distributions

## Implementation Notes

- **Particle filtering** uses systematic resampling for efficiency
- **Belief propagation** is simplified compared to full MATLAB version for clarity
- **Track formation** handles variable-length tracks and pruning
- **Extent modeling** uses inverse Wishart distributions throughout
- **Performance** may differ due to different random number generation

## File Structure

```
Meyer/
├── DataGenerator.py          # Data simulation class
├── ExtendedObjectFilter.py   # Main tracking filter class  
├── utils.py                  # Helper functions
├── test_python_classes.py    # Comprehensive test script
├── example_usage.py          # Simple usage example
├── README_Python.md          # This documentation
└── [Original MATLAB files]   # Original implementation
```

## Performance Considerations

For large-scale scenarios, consider:
- Reducing `numParticles` for faster execution
- Decreasing `numOuterIterations` for simpler belief propagation
- Using smaller surveillance regions to reduce computational load
- Implementing parallelization for particle operations (not included)

## Future Enhancements

Potential improvements:
- Full belief propagation implementation matching MATLAB version
- Parallelized particle filtering
- Advanced data association algorithms
- Performance optimization with numba/cython
- Extended visualization capabilities