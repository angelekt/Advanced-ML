import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(args):
	# Get command line arguments
	walk = args.walk
	car = args.car
	example = args.example
	x0 = args.x0
	abort = False

	if walk == True:
		# Load example data or data given by user
		if example:
			T = [50,100,200,300]
			nu = [0.005,0.01,0.05,0.1]
		else:
			T = args.T
			nu = args.n
			if T is None:
				print('Please enter at least one target point in time or set to --example')
				abort = True
			if nu is None:
				print('Please enter at least one noise variance or set to --example')
				abort = True

		# If user has given reasonable data, continue
		if not abort:
			# Initialize grid for plotting
			f, axarr = plt.subplots(len(T),len(nu))

			# Loop over every target point in time
			for i,target in enumerate(T):
				# Create time vector from 1 (0) until target for looping (plotting)
				x = np.arange(start = 1, stop=target)
				x_plot = np.arange(start = 0, stop=target)

				# Loop over every noise variance
				for j,var in enumerate(nu):
					# Start at x_0
					random_walk = [x0]

					# Loop over time
					for step in x:
						# Compute optimal control and draw random noise from Gaussian, save in random_walk
						u_star = (np.tanh(random_walk[-1]/var*(target-step))-random_walk[-1])/(target-step)
						xi = np.random.normal(loc=0.0, scale=var, size=None)
						random_walk.append(u_star + xi)

					# Plot random walk and target locations
					axarr[i,j].plot(x_plot,random_walk)
					axarr[i,j].plot(x_plot,np.ones((len(random_walk),1)))
					axarr[i,j].plot(x_plot,(-1)*np.ones((len(random_walk),1)))
					axarr[i,j].set_xlabel('Timesteps ' + r'$t$')
					axarr[i,j].set_ylabel('Location ' + r'$x$')
					axarr[i,j].set_title(r'$T$' + '=' + str(T[i]) +' '  r'$\nu$'+ '='  + str(nu[j]))

			f.suptitle('Random walk with dynamics ' + r'$dx = udt+d\xi$' + '\n' + 'Optimal control is ' + r'$u*(x,t)=\frac{tanh(\frac{x}{\nu(T-t)})-x}{T-t}$' + '\n' + 'Target location at ' + r'$t=T$' + ' is ' + r'$x = \pm 1$'
				, fontsize=14)
			plt.show()
	elif car == True:
		print('car')
	else:
		print('Please choose one exercise you want to execute')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Simulate a controlled random walk in one dimension')
	parser.add_argument('--walk', dest='walk', action='store_true', help='Set to execute random walk exercise')
	parser.set_defaults(walk=False)
	parser.add_argument('--car', dest='car', action='store_true', help='Set to execute mountain car exercise')
	parser.set_defaults(car=False)
	parser.add_argument('--example', dest='example', action='store_true', help='Set to execute with example values')
	parser.set_defaults(example=False)
	parser.add_argument('--x0', type=int,default =0,
                    help='RANDOM WALK: Starting point (default = 0)')
	parser.add_argument('--T', nargs='+', type=int,
                    help='RANDOM WALK: Target point(s) in time')
	parser.add_argument('--n', nargs='+',type=float,
                    help='RANDOM WALK: Noise variance(s)')
	args = parser.parse_args()

	main(args)