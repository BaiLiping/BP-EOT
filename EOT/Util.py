import numpy as np

def get_parameters():
    # main parameters of the statistical model
    parameters = {
        "scanTime": 0.2,
        "accelerationDeviation": 1,
        "survivalProbability": 0.99,
        "meanBirths": 0.01,
        "surveillanceRegion": np.array([[-200, 200], [-200, 200]]),
        "measurementVariance": 1**2,
        "meanMeasurements": 8,
        "meanClutter": 10,
    }

    # prior distribution parameters
    mean_target_dimension = 3
    parameters["priorVelocityCovariance"] = np.diag([10**2, 10**2])
    parameters["priorExtent2"] = 100
    parameters["priorExtent1"] = np.array(
        [[mean_target_dimension, 0], [0, mean_target_dimension]]
    ) * (parameters["priorExtent2"] - 3)
    parameters["degreeFreedomPrediction"] = 20000

    # sampling parameters
    parameters["numParticles"] = 5000
    parameters["regularizationDeviation"] = 0

    # detection and pruning parameters
    parameters["detectionThreshold"] = 0.5
    parameters["thresholdPruning"] = 10 ** (-3)
    parameters["minimumTrackLength"] = 1

    # message passing parameters
    parameters["numOuterIterations"] = 2
    return parameters
