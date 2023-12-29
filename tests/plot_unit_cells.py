from os import path, makedirs

from gbox.apps import UnitCells2D


if __name__ == '__main__':
    CSD = path.dirname(__file__)
    images_dir = path.join(CSD, "_plots")
    makedirs(images_dir, exist_ok=True)
    uc_file_path = path.join(CSD, "_data", "rve_raw_data.h5")
    #
    ucs_2d = UnitCells2D(uc_file_path)
    #
    # ucs_2d.plot(images_dir, p_bar=True)
    #
    ucs_2d.get_images_array(p_bar=True, file_path=path.join(CSD, "_data", "images_array.npz"))
    ucs_2d.get_images_array(p_bar=True, file_path=path.join(CSD, "_data", "images_array.npy"))
    # print(images_array.shape)
