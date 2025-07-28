import numpy as np

class HelmholtzSolver:
    def __init__(self, lambda0: float, boundaries: list, epsilons: list, theta: float=0, polarization: str="s"):
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

        if polarization.lower() not in ["s", "p"]:
            raise ValueError("Polarization should be 's' or 'p'")
        self.polarization = polarization.lower()

        if len(self.thicknesses) != len(epsilons):
            raise ValueError("Number of layers must equal to the number of provided permittivities")
    
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
        I = np.array([[1, r], [r, 1]], dtype=complex) / t        

        return I

    def propagation_matrix(self, kz: complex, d: float):
        """Computes the propagation matrix

        Args:
            kz (complex): z-component of the wave vector, m^-1
            d (float): layer thickness, m

        Returns:
            array: 2x2 propagation matrix
        """

        P = np.array(
            [
                [np.exp(-1j * kz * d), 0], 
                [0, np.exp(1j * kz * d)]
            ], 
                dtype=complex
        )

        return P
    
    def compute_kz(self, epsilon: complex):
        """Computes z-component of the wave vector

        Args:
            epsilon (complex): dielectric permittivity

        Returns:
            complex: z-component of the wave vector, m^-1
        """

        kx = self.k0 * np.sin(self.theta)
        kz = np.sqrt(self.k0**2 * epsilon - kx**2 + 0j)

        if np.imag(kz) < 0:
            kz = -kz
        return kz
    
    def compute_coefficients(self):

        epsilons_full = [1] + self.epsilons + [1]
        kz_list = [self.compute_kz(eps) for eps in epsilons_full]

        if self.polarization == "p":
            gamma_list = [kz/eps for kz, eps in zip(kz_list, epsilons_full)]
        else:
            gamma_list = kz_list

        M = np.eye(2, dtype=complex)

        for i, d in enumerate(self.thicknesses):
            I = self.interface_matrix(gamma_list[i], gamma_list[i+1])
            M = M @ I

            P = self.propagation_matrix(kz_list[i+1], d)
            M = M @ P

        I_last = self.interface_matrix(gamma_list[-2], gamma_list[-1])
        M = M @ I_last

        total_length = self.boundaries[-1] - self.boundaries[0]
        T = np.exp(-1j * kz_list[0] * total_length) / M[0, 0]

        R = (M[1, 0] / M[0, 0]) * np.exp(2j * kz_list[0] * self.boundaries[0])

        if self.polarization == "p":
            R = -R

        R2 = np.abs(R) ** 2
        T2 = np.abs(T) ** 2
        A = 1 - R2 - T2

        return R, T, A