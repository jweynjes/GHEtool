from typing import Optional


class GroundData:
    __slots__ = 'H', 'B', 'k_s', 'Tg', 'Rb', 'N_1', 'N_2', 'flux', 'volumetric_heat_capacity', 'alpha'

    def __init__(self, h: float, b: float, k_s: float, T_g: float, R_b: float, n_1: int, n_2: int,
                 volumetric_heat_capacity: float = 2.4 * 10**6, flux: float = 0.06) -> None:
        """
        Data for storage of ground data

        :param h: Depth of boreholes [m]
        :param b: Borehole spacing [m]
        :param k_s: Ground thermal conductivity [W/m.K]
        :param T_g: Surface ground temperature [deg C]
        (this is equal to the ground temperature at infinity when no heat flux is given (default))
        :param R_b: Equivalent borehole resistance [m K/W]
        :param n_1: Width of rectangular field [#]
        :param n_2: Length of rectangular field [#]
        :param volumetric_heat_capacity: The volumetric heat capacity of the ground (J/m3K)
        :param flux: the geothermal heat flux (W/m2)
        :return: None
        """
        self.H = h  # m
        self.B = b  # m
        self.k_s = k_s  # W/mK
        self.Tg = T_g  # °C
        self.Rb = R_b  # mK/W
        self.N_1 = n_1  # #
        self.N_2 = n_2  # #
        self.volumetric_heat_capacity = volumetric_heat_capacity  # J/m3K
        self.alpha = self.k_s / self.volumetric_heat_capacity  # m2/s
        self.flux = flux  # W/m2

    def __eq__(self, other):
        if not isinstance(other, GroundData):
            return False
        for i in self.__slots__:
            if getattr(self, i) != getattr(other, i):
                return False
        return True


class FluidData:

    __slots__ = 'k_f', 'rho', 'Cp', 'mu', 'mfr'

    def __init__(self, mfr: float, k_f: float, rho: float, Cp: float, mu: float) -> None:
        """
        Data for storage of ground data

        :param mfr: Mass flow rate per borehole [kg/s]
        :param k_f: Thermal Conductivity [W/mK]
        :param rho: Density [kg/m3]
        :param Cp: Thermal capacity [J/kgK]
        :param mu: EDynamic viscosity [Pa/s]
        :return: None
        """

        self.k_f = k_f  # Thermal conductivity W/mK
        self.mfr = mfr  # Mass flow rate per borehole kg/s
        self.rho = rho  # Density kg/m3
        self.Cp = Cp    # Thermal capacity J/kgK
        self.mu = mu    # Dynamic viscosity Pa/s

    def __eq__(self, other):
        if not isinstance(other, FluidData):
            return False
        for i in self.__slots__:
            if getattr(self, i) != getattr(other, i):
                return False
        return True


class PipeData:

    __slots__ = 'r_in', 'r_out', 'k_p', 'D_s', 'r_b', 'number_of_pipes', 'epsilon', 'k_g', 'D'

    def __init__(self, k_g: float, r_in: float, r_out: float, k_p: float, D_s: float, r_b: float, number_of_pipes: int,
                 epsilon: float = 1e-6, D: float = 4) -> None:
        """
        Data for storage of ground data

        :param k_g: Grout thermal conductivity [W/mK]
        :param r_in: Inner pipe radius [m]
        :param r_out: Outer pipe radius [m]
        :param k_p: Pipe thermal conductivity [W/mK]
        :param D_s: Distance of the pipe until center [m]
        :param r_b: Borehole radius [m]
        :param number_of_pipes: Number of pipes [#] (single U-tube: 1, double U-tube:2)
        :param epsilon: Pipe roughness [m]
        :param D: burrial depth [m]
        :return: None
        """

        self.k_g = k_g                      # grout thermal conductivity W/mK
        self.r_in = r_in                    # inner pipe radius m
        self.r_out = r_out                  # outer pipe radius m
        self.k_p = k_p                      # pipe thermal conductivity W/mK
        self.D_s = D_s                      # distance of pipe until center m
        self.r_b = r_b                      # borehole radius m
        self.number_of_pipes = number_of_pipes  # number of pipes #
        self.epsilon = epsilon              # pipe roughness m
        self.D = D                          # burial depth m

    def __eq__(self, other):
        if not isinstance(other, PipeData):
            return False
        for i in self.__slots__:
            if getattr(self, i) != getattr(other, i):
                return False
        return True


class MultipleUPPipeData(PipeData):
    """Pipe Data for multiple U pipes"""

    __slots__ = PipeData.__slots__

    def __init__(self, conductivity_grout: float, inner_radius: float, outer_radius: float, conductivity_pipe: float, pipe_distance: float,
                 borehole_radius: float, number_of_pipes: int, pipe_roughness: float = 1e-6, burial_depth: float = 4) -> None:
        """
        Data for storage of pipe data \n
        :param conductivity_grout: Grout thermal conductivity [W/mK]
        :param inner_radius: Inner pipe radius [m]
        :param outer_radius: Outer pipe radius [m]
        :param conductivity_pipe: Pipe thermal conductivity [W/mK]
        :param pipe_distance: Distance of the pipe until center [m]
        :param borehole_radius: Borehole radius [m]
        :param number_of_pipes: Number of pipes [#] (single U-tube: 1, double U-tube:2)
        :param pipe_roughness: Pipe roughness [m]
        :param burial_depth: burial depth [m]
        :return: None
        """
        super().__init__(conductivity_grout, inner_radius, outer_radius, conductivity_pipe, pipe_distance, borehole_radius, number_of_pipes,
                         pipe_roughness, burial_depth)


class CoaxialPipe(PipeData):

    __slots__ = PipeData.__slots__ + ('r_in_out', 'r_out_out', 'k_o', 'epsilon_outer', 'is_annulus_inlet')

    def __init__(self, inner_radius_inner_pipe: float, outer_radius_inner_pipe: float, conductivity_inner_pipe: float,
                 inner_radius_outer_pipe: float, outer_radius_outer_pipe: float, conductivity_outer_pipe: float,
                 borehole_radius: Optional[float] = None, conductivity_grout: Optional[float] = None, inner_pipe_roughness: float = 1e-6,
                 outer_pipe_roughness: float = 1e-6, burial_depth: float = 4, is_annulus_inlet: bool = True) -> None:
        """
        Data for storage of coaxial pipe data \n
        :param inner_radius_inner_pipe: Inner pipe inner radius [m]
        :param outer_radius_inner_pipe: Inner pipe outer radius [m]
        :param conductivity_inner_pipe: Inner Pipe thermal conductivity [W/mK]
        :param inner_radius_outer_pipe: Outer pipe inner radius [m]
        :param outer_radius_outer_pipe: Outer pipe outer radius [m]
        :param conductivity_inner_pipe: Outer Pipe thermal conductivity [W/mK]
        :param borehole_radius: Borehole radius (default=outer_radius_outer_pipe) [m]
        :param conductivity_grout: Grout thermal conductivity (default=0.1) [W/mK]
        :param inner_pipe_roughness: Inner pipe roughness (default=1e-6) [m]
        :param outer_pipe_roughness: Outer pipe roughness (default=1e-6) [m]
        :param burial_depth: burial depth (default=4) [m]
        :param is_annulus_inlet: is the annulus the inlet (downwards) pipe (default = True)
        :return: None
        """
        # check if pipe data is valid
        borehole_radius = outer_radius_outer_pipe if borehole_radius is None else borehole_radius
        conductivity_grout = 0.1 if conductivity_grout is None else conductivity_grout
        super().__init__(conductivity_grout, inner_radius_inner_pipe, outer_radius_inner_pipe, conductivity_inner_pipe, 0, borehole_radius, 0,
                         inner_pipe_roughness, burial_depth)
        self.r_in_out: float = inner_radius_outer_pipe
        self.r_out_out: float = outer_radius_outer_pipe
        self.k_o: float = conductivity_outer_pipe
        self.epsilon_outer: float = outer_pipe_roughness
        self.is_annulus_inlet: bool = is_annulus_inlet
