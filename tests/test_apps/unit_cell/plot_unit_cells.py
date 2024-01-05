from os import path, makedirs

from gbox.apps import UnitCellsDataFile

if __name__ == '__main__':
    CSD = path.dirname(__file__)
    images_dir = path.join(CSD, "_plots")
    makedirs(images_dir, exist_ok=True)
    uc_data_file_path = path.join(CSD, "uc_data", "rve_raw_data_circles_periodic.h5")
    unit_cells = UnitCellsDataFile(uc_data_file_path, periodic=True).parse()

    unit_cells.plot(
        images_dir, p_bar=True,
        matrix_color="white",
        matrix_edge_color="k",
        inclusion_color="None",
        inclusion_edge_color="b",
    )
