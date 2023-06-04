import matplotlib.pyplot as plt


class Patch:
    def __init__(self, boundaries):
        self.boundaries = boundaries
        return

    def plot(self, fig_handle=None):
        if fig_handle is None:
            plt.figure()
            fig_handle = plt.gca()
        for a_boundary in self.boundaries:
            plt.plot(a_boundary)

        return
