# Trajectory Module

Trajectories define the desired motion of the multirotor in time. There are a few options for trajectory generation. `minsnap.py` is an example of trajectory generators based on waypoints like in [Minimum Snap Trajectory Generation and Control for Quadrotors](https://ieeexplore.ieee.org/document/5980409), while the others are examples of time-parameterized shapes. 

If you'd like to make your own trajectory you can reference `traj_template.py` to see the required structure. 

Currently only trajectories that output flat outputs (position and yaw) are implemented to complement the SE3 controller, but you can develop your own trajectories that can be paired with different hierarchies of controllers. 