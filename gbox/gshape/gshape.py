from ..utils import PlotMixin, np


class GShape:
    @classmethod
    def from_params(
        cls,
        positional_params: dict[str, float],
        size_params: dict[str, float],
    ) -> "GShape":
        """
        Construct a GShape from the given positional and size parameters.

        Parameters
        ----------
        positional_params : dict[str, float]
            A dictionary containing the positional parameters of the shape.
        size_params : dict[str, float]
            A dictionary containing the size parameters of the shape.
        """
        raise NotImplementedError(
            "Subclasses must implement the sample() method."
        )

    def union_of_nspheres(self):
        """
        It returns the union of n-spheres for the current shape.
        """
        raise NotImplementedError(
            "Subclasses must implement the union_of_nspheres() method."
        )


class GShape3D(GShape):
    def union_of_spheres(self):
        raise NotImplementedError(
            "Subclasses must implement the union_of_spheres() method."
        )
