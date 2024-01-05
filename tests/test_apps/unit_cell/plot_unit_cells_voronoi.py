from os import path, makedirs

from gbox.apps import UnitCellsDataFile

if __name__ == '__main__':
    CSD = path.dirname(__file__)  # Root Directory
    uc_file_path = path.join(CSD, "uc_data", "rve_raw_data_circles_periodic.h5")
    vor_plots_dir = path.join(CSD, "_plots")
    makedirs(vor_plots_dir, exist_ok=True)
    #
    # Either plot all voronoi diagrams
    UnitCellsDataFile(uc_file_path, periodic=False).parse().plot_voronoi_diagrams(
        plots_dir=vor_plots_dir,
        f_name=lambda x: f"{x}_voronoi_diagrams.png",
    )

    # or

    # plot voronoi diagram of a specific unit cell, if its tag is known

    # or

    # plot voronoi diagram directly on the UnitCell2D
