from os import path, makedirs

from gbox.apps import UnitCellsDataFile

if __name__ == '__main__':

    case_1, case_2, case_3, case_4 = False, True, True, True

    CSD = path.dirname(__file__)  # Root Directory
    uc_file_path = path.join(CSD, "uc_data", "rve_raw_data_circles_periodic.h5")
    vor_plots_dir = path.join(CSD, "_plots")
    makedirs(vor_plots_dir, exist_ok=True)
    #
    if case_1:
        # Either plot all voronoi diagrams
        UnitCellsDataFile(uc_file_path, periodic=False).parse().plot_voronoi_diagrams(
            plots_dir=vor_plots_dir,
            f_name=lambda x: f"{x}_voronoi_diagrams.png",
        )

    if case_2:
        # plot voronoi diagram of a specific unit cell, if its tag is known
        ucs = UnitCellsDataFile(uc_file_path, periodic=False).parse().values()
        a_uc = list(ucs)[0]  # UnitCell2D
        a_uc.plot_voronoi_diagram(
            file_path=path.join(vor_plots_dir, "single_vor.png"),
            ridge_plt_opt={'face_color': 'y', 'edge_color': 'k', 'linewidth': 0.5},
        )

    # or

    # plot voronoi diagram directly on the UnitCell2D
