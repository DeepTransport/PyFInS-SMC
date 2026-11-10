import numpy as np

def greedy_OED(observation_indicator, sensor2data, weights, data, qoi=None, n_design=1):
    if qoi is not None:
        raise NotImplementedError("QoI-based greedy OED is not implemented.")
    #sensor_covY = np.diagonal(np.cov(data, rowvar=False)).copy()
    weights = weights/np.sum(weights)
    mean = np.sum(weights[:, None] * data, axis=0)
    sensor_covY = np.sum(weights[:, None] * (data - mean)**2, axis=0)
    if sensor2data is not None:
        sensor_covY = np.sum(sensor_covY[sensor2data], axis=1)
    # Exclude already selected sensors
    sensor_covY[observation_indicator] = -np.inf
    print(sensor_covY, flush=True)

    n_remaining = np.sum(~observation_indicator)
    if n_remaining == 0:
        return None, None

    n_design = min(n_design, n_remaining)
    new_sensor_ind = np.argsort(sensor_covY)[-n_design:][::-1]
    if sensor2data is not None:
        new_data_ind = np.atleast_1d(sensor2data[new_sensor_ind].reshape(-1))
    else:
        new_data_ind =  np.atleast_1d(new_sensor_ind)

    return new_sensor_ind, new_data_ind
