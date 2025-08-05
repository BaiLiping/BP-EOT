# Meyer - Original MATLAB Implementation

This directory contains the original MATLAB implementation of the Extended Object Tracking algorithm by Florian Meyer (2020).

## Reference

F. Meyer and J. L. Williams, "Scalable detection and tracking of geometric extended objects," IEEE Trans. Signal Process., vol. 69, pp. 6283–6298, Oct. 2021.

## Files

### Main Files
- `main.m` - Main script that demonstrates the algorithm with a 5-target scenario
- `eotEllipticalShape.m` - Core implementation of the EOT algorithm using belief propagation
- `PL.m` - Additional utility file

### Helper Functions (`_common/`)
- `dataAssociationBP.m` - Belief propagation for data association
- `generateClutteredMeasurements.m` - Generate measurements with clutter
- `generateTracksUnknown.m` - Generate ground truth tracks
- `getPromisingNewTargets.m` - Identify promising new target candidates
- `performPrediction.m` - Prediction step for particle filtering
- `updateParticles.m` - Update particles with measurements
- `trackFormation.m` - Form continuous tracks from estimates
- Additional utility functions for matrix operations, resampling, etc.

## Usage

To run the MATLAB implementation:

```matlab
% In MATLAB, navigate to this directory
cd Meyer

% Run the main script
main
```

The script will:
1. Generate a scenario with 5 extended objects
2. Create measurements with clutter
3. Run the EOT algorithm
4. Display tracking results

## Parameters

The main parameters are set in `main.m`:
- Number of time steps: 50
- Number of targets: 5
- Number of particles: 5000
- Surveillance region: [-200, 200] × [-200, 200]
- Mean measurements per target: 8
- Mean clutter per scan: 10

## Visualization

The results are displayed using the `showResults.m` function. Press space to start the animation.