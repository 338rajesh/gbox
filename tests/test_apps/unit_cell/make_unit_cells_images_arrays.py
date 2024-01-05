from os import path, makedirs

from gbox.apps import UnitCellsDataFile

if __name__ == '__main__':
    CSD = path.dirname(__file__)  # Root Directory
    uc_data_file_path = path.join(CSD, "uc_data", "rve_raw_data_circles_periodic.h5")
    images_dir = path.join(CSD, "_plots")
    makedirs(images_dir, exist_ok=True)
    #
    unit_cells = UnitCellsDataFile(uc_data_file_path).parse()

    uc_images_array = unit_cells.get_images_array(p_bar=True, file_path=path.join(CSD, "uc_data", "images_array.npz"))
