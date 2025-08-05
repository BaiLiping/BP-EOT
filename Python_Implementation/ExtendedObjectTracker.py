import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

class ExtendedObjectTracker:
    """
    Python implementation of MATLAB-based Extended Object Tracking (EOT) pipeline.
    Methods are ports of original MATLAB functions.
    """
    def __init__(self, parameters):
        self.parameters = parameters

    def main(self, num_steps, num_targets, mean_target_dimension,
             start_radius, start_velocity, appearance_from_to):
        """
        Entry point replicating main.m
        """
        start_states, start_matrices = self.get_start_states(
            num_targets, start_radius, start_velocity)

        # Stubs: user can fill in detailed simulation logic
        target_tracks, target_extents = self.generate_tracks_unknown(
            start_states, start_matrices, appearance_from_to, num_steps)

        measurements = self.generate_cluttered_measurements(
            target_tracks, target_extents)

        estimates = self.eot_elliptical_shape(
            measurements)

        estimated_tracks, estimated_extents = self.track_formation(
            estimates)

        self.show_results(
            target_tracks, target_extents,
            estimated_tracks, estimated_extents,
            measurements)

        return estimated_tracks, estimated_extents

    def eot_elliptical_shape(self, measurements):
        raise NotImplementedError("eot_elliptical_shape is not yet implemented.")

    def perform_prediction(self, old_particles, old_existences,
                           old_extents, scan_time):
        """
        Port of performPrediction.m
        """
        # Unpack parameters
        df_pred = self.parameters['degreeFreedomPrediction']
        surv_prob = self.parameters['survivalProbability']
        acc_dev = self.parameters['accelerationDeviation']
        A, W, _, _ = self.get_transition_matrices(scan_time)

        _, num_particles, num_targets = old_particles.shape
        new_particles = np.empty_like(old_particles)
        new_existences = np.empty_like(old_existences)
        new_extents = np.empty_like(old_extents)

        # update extents via inverse-Wishart sampling
        for t in range(num_targets):
            scaled = old_extents[:, :, :, t] / df_pred
            new_extents[:, :, :, t] = self.iwishrnd_fast_vector(
                scaled, df_pred, num_particles)

        # kinematic & existence update
        for t in range(num_targets):
            new_particles[:, :, t] = (
                A @ old_particles[:, :, t] +
                W * np.sqrt(acc_dev**2) * np.random.randn(2, num_particles)
            )
            new_existences[t] = surv_prob * old_existences[t]

        return new_particles, new_existences, new_extents

    def get_promising_new_targets(self, *args, **kwargs):
        raise NotImplementedError

    def data_association_bp(self, input_da):
        da = input_da[1, :] / input_da[0, :]
        sum_da = 1 + np.sum(da)
        out = 1.0 / (sum_da - da)
        if np.any(np.isnan(out)):
            out = np.zeros_like(out)
            out[np.argmax(da)] = 1.0
        return out

    def update_particles(self, *args, **kwargs):
        raise NotImplementedError

    def iwishrnd_fast_vector(self, parameter1, parameter2, num_particles):
        # Use wishrnd as inverse Wishart sampling
        return self.wishrnd_fast_vector(parameter1, parameter2, num_particles)

    def wishrnd_fast_vector(self, parameter1, parameter2, num_particles):
        # Port of iwishrndFastVector.m
        d = np.zeros_like(parameter1)
        d[0,0,:] = np.sqrt(parameter1[0,0,:])
        d[1,0,:] = parameter1[1,0,:] / d[0,0,:]
        d[1,1,:] = np.sqrt(parameter1[1,1,:] - d[1,0,:]**2)
        if d.shape[2] == 1:
            d = np.repeat(d, num_particles, axis=2)
        r = 2 * np.random.gamma(
            (parameter2 - np.vstack([np.zeros((1,num_particles)),
                                     np.ones((1,num_particles))]))/2,
            1)
        x = np.vstack([np.sqrt(r), np.random.randn(1, num_particles)])
        detX = 1.0 / (x[0,:] * x[1,:])
        invX = np.zeros((2,2,num_particles))
        invX[0,0,:] = detX * x[1,:]
        invX[1,1,:] = detX * x[0,:]
        invX[0,1,:] = -detX * x[2,:]
        T = np.zeros_like(invX)
        T[0,0,:] = d[0,0,:]*invX[0,0,:] + d[0,1,:]*invX[1,0,:]
        T[0,1,:] = d[0,0,:]*invX[0,1,:] + d[0,1,:]*invX[1,1,:]
        T[1,0,:] = d[1,0,:]*invX[0,0,:] + d[1,1,:]*invX[1,0,:]
        T[1,1,:] = d[1,0,:]*invX[0,1,:] + d[1,1,:]*invX[1,1,:]
        a = np.zeros_like(T)
        a[0,0,:] = T[0,0,:]**2 + T[1,0,:]**2
        a[0,1,:] = T[0,0,:]*T[0,1,:] + T[1,0,:]*T[1,1,:]
        a[1,0,:] = a[0,1,:]
        a[1,1,:] = T[0,1,:]**2 + T[1,1,:]**2
        return a

    def get_start_states(self, num_targets, radius, speed):
        pe1 = self.parameters['priorExtent1']
        pe2 = self.parameters['priorExtent2']
        # sample extents
        matrices = np.zeros((2,2,num_targets))
        for t in range(num_targets):
            matrices[:,:,t] = self.iwishrnd_fast_vector(pe1, pe2, 1)[:,:,0]
        # initial states
        states = np.zeros((4, num_targets))
        if num_targets == 1:
            states[2,0] = speed
        else:
            states[:,0] = [0, radius, 0, -speed]
            step = 2*np.pi / num_targets
            for i in range(1, num_targets):
                angle = i * step
                states[:,i] = [np.sin(angle)*radius,
                               np.cos(angle)*radius,
                               -np.sin(angle)*speed,
                               -np.cos(angle)*speed]
        return states, matrices

    def get_transition_matrices(self, scan_time):
        A = np.eye(4)
        A[0,2] = scan_time
        A[1,3] = scan_time
        W = np.zeros((4,2))
        W[0,0] = 0.5*scan_time**2
        W[1,1] = 0.5*scan_time**2
        W[2,0] = scan_time
        W[3,1] = scan_time
        reducedA = A[[0,2],:][:,[0,2]]
        reducedW = W[[0,2],0]
        return A, W, reducedA, reducedW

    def get_square2_fast(self, matrixesIn):
        out = matrixesIn.copy()
        out[0,0,:] = matrixesIn[0,0,:]**2 + matrixesIn[1,0,:]**2
        out[1,0,:] = matrixesIn[1,0,:]*matrixesIn[0,0,:] + matrixesIn[1,1,:]*matrixesIn[1,0,:]
        out[0,1,:] = matrixesIn[0,0,:]*matrixesIn[0,1,:] + matrixesIn[0,1,:]*matrixesIn[1,1,:]
        out[1,1,:] = matrixesIn[1,0,:]*matrixesIn[0,1,:] + matrixesIn[1,1,:]**2
        return out

    def mvnrnd(self, mean, cov, num_samples):
        return np.random.multivariate_normal(mean, cov, size=num_samples).T

    def track_formation(self, estimates):
        """Port of trackFormation.m"""
        num_steps = len(estimates)
        labels = []
        for est in estimates:
            if est:
                labels.extend([e['label'] for e in est])
        labels = np.unique(labels)
        num_tracks = len(labels)
        tracks = np.full((4, num_steps, num_tracks), np.nan)
        extents = np.full((2,2,num_steps,num_tracks), np.nan)
        for t, est in enumerate(estimates):
            if est:
                for e in est:
                    idx = np.where(labels == e['label'])[0][0]
                    tracks[:,t,idx] = e['state']
                    extents[:,:,:,idx][:,:,t] = e['extent']
        # prune short tracks
        min_len = self.parameters['minimumTrackLength']
        keep = [np.sum(~np.isnan(tracks[0,:,i]))>=min_len for i in range(num_tracks)]
        tracks = tracks[:,:,keep]
        extents = extents[:,:,:,keep]
        return tracks, extents

    def show_results(self, true_tracks, true_extents,
                     est_tracks, est_extents, measurements,
                     axis_values=None, visualization_mode=0):
        num_steps = measurements.shape[0]
        fig, ax = plt.subplots()
        for step in range(num_steps):
            ax.clear()
            # true
            ax.plot(true_tracks[0,:step+1,0], true_tracks[1,:step+1,0], 'r-', label='True')
            # measurement
            meas = measurements[step]
            ax.scatter(meas[0], meas[1], c='k', s=10)
            # estimated
            ax.plot(est_tracks[0,:step+1,0], est_tracks[1,:step+1,0], 'b--', label='Est')
            ax.set_aspect('equal')
            if axis_values:
                ax.set_xlim(axis_values[0])
                ax.set_ylim(axis_values[1])
            plt.pause(0.01)
        plt.show()
