import open3d as o3d
import numpy as np

pcd1 = o3d.io.read_point_cloud("result.pcd")
pcd2 = o3d.io.read_point_cloud("data\helical_curve\IrregularHelix//target.pcd")
pcd_all = pcd2 + pcd1
o3d.visualization.draw_geometries([pcd_all],
                                      window_name="重叠和非重叠点",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)

pcd3 = o3d.io.read_point_cloud("data\helical_curve\IrregularHelix//result.pcd")
pcd4 = o3d.io.read_point_cloud("data\helical_curve\IrregularHelix//target.pcd")
pcd = pcd3 + pcd4
def display_inlier_outlier(cloud, m_ind):
    inlier_cloud = cloud.select_by_index(m_ind)
    outlier_cloud = cloud.select_by_index(m_ind, invert=True)

    print("Showing non overla (red) and overlap (green): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      window_name="Overlapping and non-overlapping points",
                                      width=1024, height=768,
                                      left=50, top=50,
                                      mesh_show_back_face=False)

dists = pcd.compute_point_cloud_distance(pcd_all)
dists = np.asarray(dists)
ind = np.where(dists < 0.1)[0]

display_inlier_outlier(pcd, ind)
num_in = len(ind)
num_all = len(pcd.points)
print(f"Overlap rate{num_in/num_all}")







