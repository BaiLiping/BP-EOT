# Extended Object Tracking Implementation

This repository contains implementations of Extended Object Tracking (EOT) algorithms based on the paper:

**F. Meyer and J. L. Williams, "Scalable detection and tracking of geometric extended objects," IEEE Trans. Signal Process., vol. 69, pp. 6283–6298, Oct. 2021.**

Paper: [arXiv:2103.11279v4](arXiv-2103.11279v4/)

## Repository Structure

### `/Meyer`
Original MATLAB implementation by Florian Meyer (2020).
- `main.m` - Main script to run the MATLAB implementation
- `eotEllipticalShape.m` - Core EOT algorithm implementation
- `_common/` - Helper functions for the MATLAB implementation

### `/Python_Implementation`
Python implementation of the EOT algorithms.
- Complete Python port of the MATLAB code
- Object-oriented design with separate classes for data generation and filtering
- Test scripts and examples
- See the [Python README](Python_Implementation/README.md) for detailed documentation

### `/arXiv-2103.11279v4`
Paper source files (LaTeX) and figures.

## Overview

The Extended Object Tracking algorithm enables:
- Detection and tracking of multiple extended objects with unknown shapes
- Handling of measurement uncertainty and clutter
- Belief propagation-based data association
- Particle filtering for non-linear dynamics
- Inverse Wishart modeling of object extents

## Visualization

Visualization of Meyer's result: https://www.youtube.com/watch?v=swHLoShcozw

## Getting Started

### MATLAB Implementation
1. Navigate to the `/Meyer` directory
2. Run `main.m` in MATLAB
3. The script will generate a scenario with 5 extended objects and track them over 50 time steps

### Python Implementation
1. Navigate to the `/Python_Implementation` directory
2. Install required packages: `pip install numpy scipy matplotlib`
3. Run example scripts:
   - `python test_main_replication.py` - Replicates the MATLAB main.m behavior
   - `python example_usage.py` - Simple usage example
   - `python test_fixed_filter.py` - Test with improved filter implementation

## Key Features

- **Scalable tracking**: Handles multiple closely-spaced extended objects
- **Shape estimation**: Jointly estimates object positions and elliptical shapes
- **Belief propagation**: Efficient probabilistic data association
- **No clustering required**: Avoids traditional measurement clustering approaches
- **Particle-based**: Supports non-linear dynamics and non-Gaussian distributions

## Citation

If you use this code, please cite:

```bibtex
@article{meyer2021scalable,
  title={Scalable Detection and Tracking of Geometric Extended Objects},
  author={Meyer, Florian and Williams, Jason L},
  journal={IEEE Transactions on Signal Processing},
  volume={69},
  pages={6283--6298},
  year={2021},
  publisher={IEEE}
}
```

## License

This code is provided for research purposes. Please refer to the original paper for more details.