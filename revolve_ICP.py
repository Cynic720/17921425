import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar
import time

def downsample_point_cloud(pcd, voxel_size=0.5):
    return pcd.voxel_down_sample(voxel_size)

def point_cloud_to_kdtree(pcd):
    points = np.asarray(pcd.points)
    kdtree = KDTree(points)
    return kdtree, points

def find_correspondences(source_points, kdtree, target_points, threshold):
    correspondences = []
    for i, point in enumerate(source_points):
        dist, idx = kdtree.query(point)
        if dist < threshold:
            correspondences.append([point, target_points[idx]])

    if correspondences:
        return np.array(correspondences)
    else:
        return np.empty((0, 2, 3)) 

def get_rotation_matrix_x(angle):
    """Get the rotation matrix for a counterclockwise rotation around the x-axis by the given angle in degrees."""
    angle_radians = np.radians(angle)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, cos_angle, -sin_angle, 0],
        [0, sin_angle, cos_angle, 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def apply_transformation(points, transformation_matrix):
    """Apply the given transformation matrix to the points."""
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]

def x_icp(source_correspondences,target_correspondences):
    modified_matrix_source = source_correspondences[:, 1:]
    modified_matrix_target = target_correspondences[:, 1:]
    m1 = modified_matrix_source.T
    m2 = modified_matrix_target.T
    cosm = np.sum(m1 * m2)
    m1[1] = -m1[1]
    m2[[0, 1]] = m2[[1, 0]]
    sinm = np.sum(m1 * m2)
    return sinm, cosm

def func(a, A, B):
    radians = np.radians(a)
    return A * np.sin(radians) - B * np.cos(radians)

def find_min_angle(A, B):
    result = minimize_scalar(func, bounds=(0, 360), args=(A, B), method='bounded')
    return result.x, result.fun

def compute_errors(pc1, pc2, thre):
    distances = np.linalg.norm(pc1 - pc2, axis=1)
    inliers = distances < thre
    filtered_pc1 = pc1[inliers]
    filtered_pc2 = pc2[inliers]
    rmse = np.sqrt(np.mean(np.linalg.norm(filtered_pc1 - filtered_pc2, axis=1) ** 2))
    fitness = np.sum(inliers) / len(distances)
    return fitness, rmse

pcd_source = o3d.io.read_point_cloud("data\helical_curve\IrregularHelix\source.pcd")
pcd_target = o3d.io.read_point_cloud("data\helical_curve\IrregularHelix//target.pcd")

o3d.visualization.draw_geometries([pcd_source, pcd_target], width=600, height=600)
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0],
                         [0.0, 0.0, 0.0, 1.0]])

source_points = np.asarray(pcd_source.points)
print(len(np.asarray(pcd_source.points)))

iteration_counter = 0
threshold = 1 
total_rotation = 0


start_time = time.time()
while iteration_counter < 3000:
    try:
        kdtree = KDTree(source_points)
        target_points = np.asarray(pcd_target.points)
        correspondences = find_correspondences(target_points, kdtree, source_points, threshold)

        source_correspondences = correspondences[:, 0]
        target_correspondences = correspondences[:, 1]
        print("Number of matched point pairs:", len(correspondences))

        angle_mod_360 = total_rotation % 360
        A, B = x_icp(source_correspondences, target_correspondences)
        angle, _ = find_min_angle(A, B)
        transformation_matrix = get_rotation_matrix_x(angle)
        print("Rotation angle：", angle_mod_360)

        pcd_source.points = o3d.utility.Vector3dVector(
            apply_transformation(source_points, transformation_matrix))
        source_points = np.asarray(pcd_source.points)

        total_rotation += angle
        iteration_counter += 1

        if iteration_counter % 10 == 0:
            threshold = 0.4 * (1 + np.cos(np.pi * iteration_counter / 3000)) + 0.2
            print(f"The number of iterations {iteration_counter}: The threshold is reduced to {threshold}")
    except IndexError as e:
        print(f"IndexError: {e}, Continue with the subsequent code")
        break 


print("The angle at which the function is minimized:", total_rotation)
end_time = time.time()
total_time = end_time - start_time
print(f"Program running time: {total_time:.2f} sec")
pcd_source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))


pcd_source.paint_uniform_color([1, 0, 0])
pcd_target.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pcd_source, pcd_target], window_name="Algorithm registration",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)

trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0],
                             [0.0, 0.0, 0.0, 1.0]])
evaluation = o3d.pipelines.registration.evaluate_registration(pcd_source, pcd_target, threshold, trans_init)
print(evaluation)
o3d.io.write_point_cloud("result.pcd", pcd_source)




