# TracTrac PTV Software: a fast multi-object tracking algorithm for motion estimation

TracTrac is an open-source Matlab/Python implementation of a robust and efficient object tracking algorithm capable of simultaneously tracking several thousands of objects in very short time. Originally developed as an alternative to particle image velocimetry algorithms for estimating fluid flow velocities, its versatility and robustness makes it relevant to many other dynamic sceneries encountered in geophysics such as granular flows and drone videography. In this article, the structure of the algorithm is detailed and its capacity to resolve strongly variable and intermittent object motions is tested against three examples of geophysical interest.
TracTrac is a Particle Tracking Velocimetry (PTV) software which is fast (more than 10000 points tracked per second) and accurate (up to 0.01 pixel resolution), forming thus a good concurrent to the state-of-the art PIV algorithms. It allows to track anything that moves: birds, ants, grains, water flows... It runs on Python (v2.7 with OpenCV2) or Matlab (>2012a with Image Processing toolbox). Give it a try !


# Reference
Please cite the following reference:

Heyman J., TracTrac: A fast multi-object tracking algorithm for motion estimation, Computers & Geosciences, Volume 128, 2019, Pages 11-18,doi: 10.1016/j.cageo.2019.03.007

The article is available in <a href="https://perso.univ-rennes1.fr/joris.heyman/PDF/tractrac_final.pdf" > my personal webpage </a>

Contact: joris.heyman@univ-rennes1.fr 


# User guide

Matlab users, run gui with 
> tractrac 

Python users, run test cases with
> python tractrac.py -p 1
or
> python tractrac.py -f '../Sample_videos/videotest.avi' -p 1
or
> python tractrac.py -f '../Sample_videos/PIVChallenge/*.tif' -p 2
or
> python tractrac.py -f '../Sample_videos/RiverDrone/*.tif' -p 2

Run live-treatment by webcam acquisition by
> python tractrac.py -f '0' -p 1 -cmax 50

Details of parameters and commands are given in the publication <a href="https://perso.univ-rennes1.fr/joris.heyman/PDF/tractrac_final.pdf" > Heyman (2019) </a>.

