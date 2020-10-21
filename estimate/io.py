import h5py


class Hdf5Io():

    def read_point_cloud_dataset(self, filepath, group, dataset):

        with h5py.File(filepath, 'r') as f:
            group = f[group]
            dset = group[dataset]
            return dset[:]

        return []
        # points = dset[:, 0:3]
        # normals = dset[:, 3:6]
        # radians = dset[:, 6]

    def write_point_cloud_dataset_with_gradient_average(self, filepath,
                                                        npcd, timestamp,
                                                        group, dataset):
        is_success = False
        with h5py.File(filepath, 'a') as fw:

            group = fw.create_group(f'/{group}')

            dset = group.create_dataset(
                name=dataset,
                data=npcd,
            )

            dset.attrs['created_at'] = timestamp
            is_success = True

        return is_success


if __name__ == '__main__':

    hdf5 = Hdf5Io()

    path = '/home/latona/comet/Data/calculate-vectors-gradients/file/output/20200616192455470.hdf5'
    dset = hdf5.read_point_cloud_dataset(
        path, 'point-cloud', 'calculate-vectors-gradients')

    print(dset)
    print(dset.shape)
    print(type(dset))
