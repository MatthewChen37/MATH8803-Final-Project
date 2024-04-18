import numpy as np

# Just like the Simulations! https://star-wars-memes.fandom.com/wiki/Just_like_the_simulations!

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

# Classic Henon System implementation
# For single channels
def henon_system2(x0, T):
	x = np.zeros(T + 1)
	y = np.zeros(T + 1)

	a = 1.4
	b = 0.3


	# Initial Conditions
	x[0] = x0[0]
	y[0] = x0[1]

	for t in range (1, T):
		x[t] = 1 - a * (x[t - 1] ** 2) + y[t - 1]
		y[t] = b * x[t - 1]

	return x, y

# One Dimensional Henon map decomposition as described from Wikipedia:
# https://en.wikipedia.org/wiki/H%C3%A9non_map
def henon_system3(x0, T):
	x = np.zeros(T + 1)
	a = 1.4
	b = 0.3

	# Initial Conditions
	x[0] = x0[0]
	x[1] = x0[0] # assume initial x is same for first two time steps

	for t in range (2, T):
		x[t] = 1 - a * (x[t - 1] ** 2) + b * x[t - 2]
	return x

# simulate Rossler system using Forward Euler implementation
def rossler_system_1(x0, params, T, timestep):
	x = np.zeros(T)
	y = np.zeros(T)
	z = np.zeros(T)

	x[0] = x0[0]
	y[0] = x0[1]
	z[0] = x0[2]

	a = params[0]
	b = params[1]
	c = params[2]

	for t in range(1, T):
		dxdt = -y[t - 1] - z[t - 1]
		dydt = x[t - 1] + a * y[t - 1]
		dzdt = b + z[t - 1] * (x[t - 1] - c)

		x[t] = x[t - 1] + timestep * dxdt
		y[t] = y[t - 1] + timestep * dydt
		z[t] = z[t - 1] + timestep * dzdt
	return x, y, z

# simulate Lorenz system using Forward Euler implementation
def lorenz_system_1(x0, params, T, timestep):
	x = np.zeros(T + 1)
	y = np.zeros(T + 1)
	z = np.zeros(T + 1)

	x[0] = x0[0]
	y[0] = x0[1]
	z[0] = x0[2]

	sigma = params[0]
	beta = params[1]
	rho = params[2]


	for t in range(1, T + 1):
		dxdt = sigma * (y[t - 1] - x[t - 1])
		dydt = x[t - 1] * (rho - z[t - 1]) - y[t - 1]
		dzdt = (x[t - 1] * y[t - 1]) - (beta * z[t - 1])
		x[t] = x[t - 1] + timestep * dxdt
		y[t] = y[t - 1] + timestep * dydt
		z[t] = z[t - 1] + timestep * dzdt
	return x, y, z





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