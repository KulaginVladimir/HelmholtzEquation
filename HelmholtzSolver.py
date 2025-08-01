import numpy as np


class HelmholtzSolver:
    def __init__(
        self,
        lambda0: float,
        boundaries: list,
        epsilons: list,
        theta: float = 0,
        polarization: str = "s",
    ):
        """
        Args:
            lambda0 (float): wavelength in vacuum
            boundaries (list): list of structure layers
            epsilons (list): list determining the piecewise dielectric permittivity
            theta (float, optional): angle of incidence, deg. Defaults to 0.
            polarization (str, optional): wave polarizations. Defaults to "s".
        """

        self.lambda0 = lambda0
        self.k0 = 2 * np.pi / lambda0
        self.boundaries = boundaries
        self.thicknesses = np.diff(boundaries)
        self.epsilons = epsilons
        self.theta = np.deg2rad(theta)
        self.M = np.eye(2, dtype=np.complex256)

        if polarization.lower() not in ["s", "p"]:
            raise ValueError("Polarization should be 's' or 'p'")
        self.polarization = polarization.lower()

        if len(self.thicknesses) != len(epsilons):
            raise ValueError(
                "Number of layers must equal to the number of provided permittivities"
            )

        if any(self.thicknesses) < 0:
            raise ValueError("Layer positions should be ascending")

    def interface_matrix(self, gamma_left: complex, gamma_right: complex):
        """Computes the interface matrix

        Args:
            gamma_left (complex): k (or k/epsilon) at the left, m^-1
            gamma_right (complex): k (or k/epsilon) at the right, m^-1

        Returns:
            array: 2x2 interface matrix
        """

        r = (gamma_left - gamma_right) / (gamma_left + gamma_right)
        t = 2 * gamma_left / (gamma_left + gamma_right)
        I = np.array([[1, r], [r, 1]], dtype=np.complex256) / t

        return I

    def propagation_matrix(self, kz: complex, d: float):
        """Computes the propagation matrix

        Args:
            kz (complex): z-component of the wave vector, m^-1
            d (float): layer thickness, m

        Returns:
            array: 2x2 propagation matrix
        """

        z1 = -1j * kz * d
        z2 = 1j * kz * d
        phi1 = np.exp(z1.real, dtype=np.float128) * (
            np.cos(z1.imag, dtype=np.float128) + 1j * np.sin(z1.imag, dtype=np.float128)
        )
        phi2 = np.exp(z2.real, dtype=np.float128) * (
            np.cos(z2.imag, dtype=np.float128) + 1j * np.sin(z2.imag, dtype=np.float128)
        )

        P = np.array([[phi1, 0], [0, phi2]], dtype=np.complex256)

        return P

    def compute_kz(self, epsilon: complex):
        """Computes z-component of the wave vector

        Args:
            epsilon (complex): dielectric permittivity

        Returns:
            complex: z-component of the wave vector, m^-1
        """

        kx = self.k0 * np.sin(self.theta, dtype=np.complex256)
        kz = np.sqrt(self.k0**2 * epsilon - kx**2 + 0j, dtype=np.complex256)

        if np.imag(kz) < 0:
            kz = -kz
        return kz

    def T(self):
        """Computes amplitude transmission coefficient

        Returns:
            complex: amplitude transmission coefficient
        """

        return 1 / self.M[0, 0]

    def R(self):
        """Computes amplitude reflection coefficient

        Returns:
            complex: amplitude reflection coefficient
        """

        return self.M[1, 0] / self.M[0, 0]

    def compute_coefficients(self):

        epsilons_full = [1] + self.epsilons + [1]
        kz_list = [self.compute_kz(eps) for eps in epsilons_full]

        if self.polarization == "p":
            gamma_list = [kz / eps for kz, eps in zip(kz_list, epsilons_full)]
        else:
            gamma_list = kz_list

        n_layers = len(self.thicknesses)
        cut_off = False

        for i in range(n_layers):
            I = self.interface_matrix(gamma_list[i], gamma_list[i + 1])
            self.M = self.M @ I

            # Check for complete absorption in the next layer
            kz = kz_list[i + 1]
            d = self.thicknesses[i]
            if kz.imag * d > 50:
                cut_off = True
                break

            P = self.propagation_matrix(kz, d)
            self.M = self.M @ P

        if not cut_off:
            I_last = self.interface_matrix(gamma_list[-2], gamma_list[-1])
            self.M = self.M @ I_last

            T = self.T()
        else:
            T = 0

        R = self.R()

        if self.polarization == "p":
            R = -R

        R2 = np.abs(R) ** 2
        T2 = np.abs(T) ** 2
        A = 1 - R2 - T2

        return R, T, A
