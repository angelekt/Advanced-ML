## Overview

When you run a computation, it will show all respective plot by default. Please follow the upcoming steps to run.

1. Use the 'config.yaml' file to setup the run.
2. Select between the number of joints 'n_joints'.
3. Set other hyperparameters


## Possible configurations to reproduce the plots:

* Figure 1 + 2
`n_joints_3: 3, n_joints_100: 100, joint_length: 1,  dt: 0.001, T: 2, lr: 0.0001 , alpha: 100, elbo: True, Example1: True, target_flag: True `
* Figure 4
`n_joints_3: 3, n_joints_100: 100, joint_length: 1,  dt: 0.001, T: 2, lr: 0.0001 , alpha: 0.1, elbo: True, Example1: True, target_flag: True `
* Figure  3
`n_joints_3: 3, n_joints_100: 100, joint_length: 1,  dt: 0.005, T: 2, lr: 0.0001 , alpha: 0.1, elbo: True, Example1: False, Example2: True, target_flag: False`

If you want to reproduce your own random results set your parameters and:
`Example1: False, Example2: False`

## Link to Overleaf Report
https://www.overleaf.com/read/tyjsbphyqjwx
