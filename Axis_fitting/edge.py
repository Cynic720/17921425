import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def read_pcd(filename):
    """Read a PCD file and return the points as a numpy array."""
    pcd = o3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points)

def project_points_to_plane(points, normal):
    """Project points onto a plane defined by its normal vector."""
    normal = normal / np.linalg.norm(normal)
    projection_matrix = np.eye(3) - np.outer(normal, normal)
    projected_points = points @ projection_matrix.T
    return projected_points

def fit_circle_2d(points):
    """Fit a circle to 2D points."""
    def residuals(c, x, y):
        return np.sqrt((x - c[0])**2 + (y - c[1])**2) - c[2]

    x_2d, y_2d = points[:, 0], points[:, 1]
    c0 = np.array([np.mean(x_2d), np.mean(y_2d), 1])
    res = least_squares(residuals, c0, args=(x_2d, y_2d))
    circle_center_2d, circle_radius = res.x[:2], res.x[2]
    return circle_center_2d, circle_radius

pcd = o3d.io.read_point_cloud("data.pcd")
pcd = pcd.voxel_down_sample(voxel_size=1)
print(pcd)  
o3d.visualization.draw_geometries([pcd], window_name="Raw point cloud",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)
print("Radius oulier removal")
cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=3)
radius_cloud = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([radius_cloud], window_name="Radius filtering",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)

points = np.asarray(radius_cloud.points)
plane_normal = np.array([1.0, 0.0, 0.0])  # Axis direction

projected_points = project_points_to_plane(points, plane_normal)

basis_1 = np.array([plane_normal[1], -plane_normal[0], 0])
if np.linalg.norm(basis_1) == 0:
    basis_1 = np.array([0, plane_normal[2], -plane_normal[1]])
basis_1 /= np.linalg.norm(basis_1)
basis_2 = np.cross(plane_normal, basis_1)
basis_2 /= np.linalg.norm(basis_2)

basis_matrix = np.vstack([basis_1, basis_2]).T
projected_2d_points = projected_points @ basis_matrix

circle_center_2d, circle_radius = fit_circle_2d(projected_2d_points)

circle_center_3d = circle_center_2d @ basis_matrix.T 

plt.figure()
plt.scatter(projected_2d_points[:, 0], projected_2d_points[:, 1], label='Projected Points')
theta_fit = np.linspace(0, 2 * np.pi, 100)
circle_points_2d = np.array([
    circle_center_2d[0] + circle_radius * np.cos(theta_fit),
    circle_center_2d[1] + circle_radius * np.sin(theta_fit)
]).T

plt.plot(circle_points_2d[:, 0], circle_points_2d[:, 1], 'r-', label='Fitted Circle')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Projected Points and Fitted Circle')
plt.show()

print(f"Circle center (2D plane): {circle_center_2d}")
print(f"Circle radius: {circle_radius}")
print(f"Circle center (3D space): {circle_center_3d}")
