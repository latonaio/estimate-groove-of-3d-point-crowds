import os
import sys

import aion.common_library as common
import cupoch as cph
import numpy as np
import open3d as o3d
from StatusJsonPythonModule.StatusJsonRest import StatusJsonRest

from estimate.groove import estimate_groove_index, is_grooves
from estimate.io import Hdf5Io

OUTPUTDIR = common.get_output_path(os.getcwd(), __file__)

TEST = False


def main():

    status_obj = StatusJsonRest(os.getcwd(), __file__)
    status_obj.initializeInputStatusJson()

    # NOTE: FOR TEST and need to fix
    if not TEST:
        pointcloud = status_obj.getMetadataFromJson(
            'calculate-gradient-averages')
        path = pointcloud['filepath']
        group = pointcloud['group']
        r_dset_name = pointcloud['dataset']
        timestamp = pointcloud['timestamp']
    else:
        # path = "/home/latona/comet/Data/calculate-gradient-averages/file/output/20200615160546791.hdf5"
        path = "/home/latona/comet/Data/calculate-gradient-averages/file/output/20200617204904424.hdf5"
        group = "point-cloud"
        r_dset_name = "calculate-gradient-averages"

    hdf5io = Hdf5Io()
    dst = hdf5io.read_point_cloud_dataset(path, group, r_dset_name)
    gradients = dst[:, -1]
    _candinates = is_grooves(gradients)
    groove_candinates = _candinates.reshape(_candinates.shape[0], 1)

    _dst = np.block([dst[:], groove_candinates])

    if TEST:
        print(dst.shape)
        idx = estimate_groove_index(dst)
        print(len(idx))
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dst[:, 0:3])
        pcd.normals = o3d.utility.Vector3dVector(dst[:, 3:6])
        pcd.paint_uniform_color([0., 0, 1])
        np.asarray(pcd.colors)[idx[:], :] = [1, 0, 0]
        o3d.visualization.draw_geometries([pcd])
        """
        pcd = cph.geometry.PointCloud()
        pcd.points = cph.utility.Vector3fVector(dst[:, 0:3])
        pcd.normals = cph.utility.Vector3fVector(dst[:, 3:6])
        pcd.paint_uniform_color([0., 0., 1])
        colors = np.asarray(pcd.colors.cpu())
        colors[idx[:], :] = [1, 0, 0]
        pcd.colors = cph.utility.Vector3fVector(colors)
        downpcd = pcd.voxel_down_sample(1)
        downpcd.normals = cph.utility.Vector3fVector(dst[:, 3:6])

        cph.visualization.draw_geometries([pcd])

    if not TEST:
        output_path = output_path(OUTPUTDIR, f'{timestamp}.hdf')
        hdf5io.write_point_cloud_dataset_with_gradient_average(
            dset, output_path, group, w_dset_name
        )


if __name__ == '__main__':
    main()
