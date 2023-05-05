# pysizeunfolder
A python library with various functions related to stereological methods. Currently, there are some functions for taking random line sections of convex polygons (2D) and plane sections of convex polyhedrons (3D). These random sections are known as Isotropic Uniformly Random (IUR) sections. There are functions for estimating 3D grain/ particle size distributions from 2D sections. The details of this implementation, and the mathematical justification are given in the papers listed in the section References.

## Table of contents
* [Installation and dependencies](#installation-and-dependencies)
* [Code examples: Random sections of polygons](#code-examples-random-sections-of-polygons)
* [Code examples: Random sections of polyhedrons](#code-examples-random-sections-of-polyhedrons)
* [Code examples: Estimating particle size distributions](#code-examples-estimating-particle-size-distributions)
* [Documentation](#documentation)
* [References](#references)

## Installation and dependencies
The library may be installed by running:

```
pip install git+https://github.com/thomasvdj/vorostereology
```

If you do not have git installed and you use the popular python Anaconda distribution, then you may install git with: 

```
conda install git
```

This package depends on Numpy, Scipy, Cython and KDEpy, these are installed automatically.

## Code examples: Random sections of polygons
To highlight some of the functionalities of this library we present some code snippets. All of the examples below can be found in more detail in scripts in the examples folder. The following imports are used throughout the examples:
```
import pysizeunfolder as pu
import numpy as np
```
### Example 1
In the following example we define the centered unit square via its vertices. Then, we take 1 million random line sections of the square and we obtain the lengths. Given this sample of lengths we approximate the associated probability density function of this distribution. The random generator rng is used to make the results reproducible.
```
rng = np.random.default_rng(0)
points = np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]])
lengths = pu.iur_2d_hull(points, n=1000000, rng)
x, y = pu.approx_length_density(lengths)
```
Plotting lengths in a histogram and the x, y points of the approximation with matplotlib:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/square_estimate.png" width=50% height=50%>

### Example 2
In the following example we generate 10 random points in the centered unit square, and we define a polygon as the convex hull of these points. For this given polygon we take 100 random sections, and we also retrieve the vertices of the corresponding lines.
```
points = rng.uniform(low=-0.5, high=0.5, size=(10, 2))
lengths, vertices = pu.iur_2d_hull(points, n=100, return_vertices=True)
```
Visualizing the polygon and the linear sections with matplotlib:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/random_polygon.png" width=50% height=50%>

## Code examples: Random sections of polyhedrons

### Example 1
In the following example we define the centered unit cube via its vertices. Then, we take 1 million random plane sections of the cube and we obtain the areas. Given this sample of areas we approximate the associated probability density function of this distribution. 
```
points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                   [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
areas = pu.iur_3d_hull(points, n=1000000, rng=rng)
x, y = pu.approx_area_density(areas)
```
Plotting areas in a histogram and the x, y points of the approximation with matplotlib:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/cube_estimate.png" width=50% height=50%>

### Example 2
In the following example we generate 15 random points in the centered unit cube, and we define a polyhedron as the convex hull of these points. For this given polyhedron we take a single random plane section, and we also retrieve the vertices of the resulting polygon.
```
points = rng.uniform(low=-0.5, high=0.5, size=(15, 3))
area, section = pu.iur_3d_hull(points, 1, return_vertices=True, rng=rng)
```
Visualizing the polyhedron and the planar section with pyvista:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/random_polyhedron.png" width=50% height=50%>

Note: for the 3D visualization pyvista was used, which is not installed by default. It may be installed via: 

```
pip install vtk pyvista
```

## Code examples: Estimating particle size distributions
Suppose we have a sample of observed 2D section areas. This sample of section areas is the result of intersecting a system of randomly positioned and oriented cubes with a plane, the sizes of the cubes vary. We wish to estimate the size disribution of the cubes. In this particular case, the size distribution corresponds to the distribution of edge lengths of the 3D cubes.

First, run the file parallel.py in the example folder. This esssentially runs Example 2 (random sections of the centered unit cube), via a parallel implementation such that we can quickly obtain a large sample of random sections of the unit cube. This large sample, which we call reference_sample is used by the estimation procedure. Below we also generate a sample of n=1000 observed section areas, which corresponds to a standard exponential size distribution of the 3D cubes.
```
rng = np.random.default_rng(0)
reference_sample = pickle.load(open("cube_sample.pkl", "rb"))
n = 1000
sizes = rng.gamma(shape=2, scale=1, size=n)
areas = pu.iur_3d_shape("cube", n, rng=rng)
sample = np.square(sizes)*areas
x_pts, y_pts = pu.estimate_size(sample, reference_sample)
```
Visualizing the estimated distribution function and the true distribution function:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/size_estimate.png" width=50% height=50%>

Keep in mind that in this theoretical setting the underlying 3D size distribution is known, so we can verify how close the estimate is to the truth.

In the estimation procedure, we first estimate to so-called length-biased size distribution. This distribution may be somewhat more difficult to interpret, compared to the size distribution. Still, if you wish to skip the additional de-biasing step you may set debias=False as follows:
```
x_pts, y_pts = pu.estimate_size(sample, reference_sample, debias=False)
```
Visualizing the estimated length-biased distribution function and the true length-biased distribution function:

<img src="https://github.com/thomasvdj/pysizeunfolder/blob/main/examples/biased_size_estimate.png" width=50% height=50%>

This estimation procedure can be performed in principle for any choice of shape for the particles. This can be done by providing an appropriate reference sample corresponding to this chosen shape. The sample may be generated by pu.iur_3d_hull, pu.iur_3d_half or pu.iur_3d_shape. It is then wise to save this reference sample to a file, for later use, as it may take a few minutes to generate such a (very large) sample.

## Documentation
For now the documentation consists of the code examples as above (and in the examples folder). Each function in this library is also documented using docstrings, which can be found in the source code.

## References
If you find this code useful, or are interested in the mathematical details we refer to the following papers:
```
@article{vdjagt2023_1,
  title={Existence and approximation of densities of chord length- and cross section area distributions},
  author={Thomas van der Jagt and Geurt Jongbloed and Martina Vittorietti},
  journal={arXiv preprint arXiv:2305.02864},
  year={2023}
}
```
```
@article{vdjagt2023_2,
  title={Stereological determination of particle size distributions for similar convex bodies},
  author={Thomas van der Jagt and Geurt Jongbloed and Martina Vittorietti},
  journal={arXiv preprint arXiv:2305.02856},
  year={2023}
}
```