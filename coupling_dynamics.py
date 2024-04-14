import numpy as np

# From Supplementary Material: A systematic framework for functional connectivity measures section 1.5
def henon_system(x0, Cij, T):
	np.random.seed(seed=123456789)
	channels = x0.shape[0]


	x_t = np.zeros((channels, T))
	x_t[:, 0] = x0
	x_t[:, 1] = x0


	a = 1.4
	b = 0.3

	noise = np.zeros((channels, T))

	for timestep in range(2, T):
		for channel_i in range(channels):
			sum1 = 0 
			for channel_j in range(channels):
				sum1 += Cij[channel_i, channel_j] * x_t[channel_i, timestep - 1] * x_t[channel_j, timestep - 1]
			sum2 = 0
			for channel_j in range(channels):
				sum2 += Cij[channel_i, channel_j]
			noise[channel_i, timestep] = np.random.normal(loc=0.0, scale=1.0, size=None)
			x_t[channel_i, timestep] = a - sum1 - ((1 - sum2) * x_t[channel_i, timestep - 1]**2) + b * x_t[channel_i, timestep - 2] #+ noise[channel_i, timestep]
	return x_t

def mean_squared_error(Cij, x_real, dynamic_system):
	channels = x_real.shape[0]
	Cij = Cij.reshape((channels, channels))

	assert x_real.shape[0] == Cij.shape[0]
	assert x_real.shape[0] == Cij.shape[1]

	x_real -= x_real[:, 0]
	T = x_real.shape[1]
	x_system = dynamic_system(x_real[:, 0], Cij, T)
	assert x_real.shape == x_system.shape
	return (np.square(x_real - x_system)).mean()