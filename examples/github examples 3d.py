import pysizeunfolder as pu
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv  # Install via: pip install vtk pyvista


# Author: Thomas van der Jagt


# Example 1

rng = np.random.default_rng(0)
points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                   [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
areas = pu.iur_3d_hull(points, n=1000000, rng=rng)
x, y = pu.approx_area_density(areas)

plt.figure(figsize=(4, 3))
plt.hist(areas, bins=80, ec='black', linewidth=0.2, density=True)
plt.plot(x, y)
plt.xlim(0, 1.5)
plt.ylim(0, 3)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.14)
plt.xlabel("Area")
plt.ylabel("Density")
plt.savefig("cube estimate.png", dpi=600)
#plt.show()

# Example 2

rng = np.random.default_rng(2)
points = rng.uniform(low=-0.5, high=0.5, size=(15, 3))
area, section = pu.iur_3d_hull(points, 1, return_vertices=True, rng=rng)
faces = pu.faces_3d_hull(points)

faces = np.hstack([[len(face)] + face for face in faces])
polygon_mesh = pv.PolyData(points, faces)
section_mesh = pv.PolyData(section[0], [len(section[0])] + list(range(len(section[0]))))

pv.set_plot_theme('document')
p = pv.Plotter(window_size=(1000, 1000))
p.add_mesh(polygon_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:blue",
           opacity=0.35, line_width=2)
p.add_mesh(section_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:red",
           line_width=2)
p.camera.elevation = -15
p.camera.azimuth = 5
p.camera.zoom(1.3)
p.enable_anti_aliasing()
p.show(screenshot="random polyhedron.png")

# Various code examples

areas = pu.iur_3d_shape("dodecahedron", n=10000)
x, y = pu.approx_area_density(np.sqrt(areas), sqrt_data=True)
# This is a halfspace representation of the centered unit cube
halfspaces = np.array([[0, 0, 1, -0.5], [0, 0, -1, -0.5], [1, 0, 0, -0.5],
                       [-1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, -1, 0, -0.5]], dtype=np.double)
origin = np.array([0, 0, 0], dtype=np.double)
areas, sections = pu.iur_3d_half(halfspaces, origin, n=10, return_vertices=True, rng=rng)
