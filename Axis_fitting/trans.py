import numpy as np
import open3d as o3d


def read_pcd(filename):
    """Read a PCD file and return the points as a numpy array."""
    pcd = o3d.io.read_point_cloud(filename)
    return np.asarray(pcd.points), pcd


def get_transformation_matrix(start_point, direction):
    direction = direction / np.linalg.norm(direction)

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -start_point

    x_axis = np.array([1, 0, 0])
    v = np.cross(direction, x_axis)
    s = np.linalg.norm(v)
    c = np.dot(direction, x_axis)

    if s == 0:  
        rotation_matrix = np.eye(4)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_3x3 = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation_3x3

    transformation_matrix = np.dot(rotation_matrix, translation_matrix)

    return transformation_matrix


def apply_transformation(points, transformation_matrix):
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]


def save_point_cloud(points, filename):
    """Save the given points as a PCD file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


input_filename = 'data.pcd'
points, original_pcd = read_pcd(input_filename)

start_point = np.array([  0.0,         -16.67711227,  88.6613555 ])  # Center
direction = np.array([1.0, 0.0, 0.0])  # PCA Axis direction

transformation_matrix = get_transformation_matrix(start_point, direction)

transformed_points = apply_transformation(points, transformation_matrix)

output_filename = 'data.pcd'
save_point_cloud(transformed_points, output_filename)

print("Transformation Matrix:")
print(transformation_matrix)

