import numpy as np
import matplotlib.pyplot as plt

class GaussianBeam:
    def __init__(
        self,
        N,
        size_screen,
        w0,
        x_0=0.0,
        y_0=0.0,
        amplitude=1.0,
        phase=None
    ):
        self.N = N
        self.size_screen = size_screen
        self.w0 = w0
        self.x_0 = x_0
        self.y_0 = y_0
        self.amplitude = amplitude

        self.dx = size_screen / N

        x = (np.arange(N) - N // 2) * self.dx

        self.xx, self.yy = np.meshgrid(
            x,
            x,
            indexing="xy"
        )

        r2 = (
            (self.xx - x_0) ** 2
            + (self.yy - y_0) ** 2
        )

        self.E = amplitude * np.exp(
            -r2 / w0**2
        ).astype(complex)

        if phase is not None:
            self.apply_phase(phase)

    def get_xy(self):
        return self.xx, self.yy

    def get_field(self):
        return self.E.copy()

    def get_intensity(self, normalized=False):
        intensity = np.abs(self.E) ** 2

        if normalized and intensity.max() > 0:
            intensity = intensity / intensity.max()

        return intensity

    def get_phase(self):
        return np.angle(self.E)

    def get_power(self):
        return np.sum(
            np.abs(self.E) ** 2
        ) * self.dx**2

    def normalize_power(self, power=1.0):
        current_power = self.get_power()

        if current_power > 0:
            self.E *= np.sqrt(
                power / current_power
            )

        return self.E

    def apply_phase(self, phase):
        phase = np.asarray(phase)

        if phase.shape != self.E.shape:
            raise ValueError(
                f"phase должна иметь форму {self.E.shape}"
            )

        self.E *= np.exp(1j * phase)

        return self.E

    def apply_amplitude_mask(self, mask):
        mask = np.asarray(mask)

        if mask.shape != self.E.shape:
            raise ValueError(
                f"mask должна иметь форму {self.E.shape}"
            )

        self.E *= mask

        return self.E

    def get_circular_aperture(self, diameter):
        mask = Propagator.get_mask(
            self.N,
            self.size_screen,
            diameter
        )

        return self.E * mask

    def apply_circular_aperture(self, diameter):
        self.E = self.get_circular_aperture(
            diameter
        )

        return self.E

    def reset(
        self,
        x_0=None,
        y_0=None,
        w0=None,
        amplitude=None
    ):
        if x_0 is not None:
            self.x_0 = x_0

        if y_0 is not None:
            self.y_0 = y_0

        if w0 is not None:
            self.w0 = w0

        if amplitude is not None:
            self.amplitude = amplitude

        r2 = (
            (self.xx - self.x_0) ** 2
            + (self.yy - self.y_0) ** 2
        )

        self.E = (
            self.amplitude
            * np.exp(-r2 / self.w0**2)
        ).astype(complex)

        return self.E

    def show_intensity(
        self,
        diameter=None,
        normalized=False
    ):
        if diameter is None:
            E = self.E
        else:
            E = self.get_circular_aperture(
                diameter
            )

        intensity = np.abs(E) ** 2

        if normalized and intensity.max() > 0:
            intensity = intensity / intensity.max()

        extent = [
            -self.size_screen / 2,
            self.size_screen / 2,
            -self.size_screen / 2,
            self.size_screen / 2
        ]

        plt.figure(figsize=(8, 7))
        image = plt.imshow(
            intensity,
            extent=extent,
            origin="lower",
            cmap="hot"
        )
        plt.xlabel("x, м")
        plt.ylabel("y, м")
        plt.colorbar(image)
        plt.show()

    def show_phase(self, diameter=None):
        phase = np.angle(self.E)

        if diameter is not None:
            mask = Propagator.get_mask(
                self.N,
                self.size_screen,
                diameter
            )

            phase = np.where(
                mask > 0,
                phase,
                np.nan
            )

        extent = [
            -self.size_screen / 2,
            self.size_screen / 2,
            -self.size_screen / 2,
            self.size_screen / 2
        ]

        plt.figure(figsize=(8, 7))
        image = plt.imshow(
            phase,
            extent=extent,
            origin="lower",
            cmap="twilight"
        )
        plt.xlabel("x, м")
        plt.ylabel("y, м")
        plt.colorbar(image)
        plt.show()

    def show_gaussian_beam(
        self,
        normalized=False
    ):
        intensity = self.get_intensity(
            normalized=normalized
        )

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.plot_surface(
            self.xx,
            self.yy,
            intensity
        )

        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
        ax.set_zlabel("Интенсивность")

        plt.show()

    def show_gaussin_beam(
        self,
        normalized=False
    ):
        self.show_gaussian_beam(
            normalized=normalized
        )

class Propagator:
    @staticmethod
    def get_mask(N, size_screen, diameter, center=(0.0, 0.0)):
        dx = size_screen / N
        x = (np.arange(N) - N // 2) * dx
        X, Y = np.meshgrid(x, x, indexing="xy")

        x0, y0 = center
        R = diameter / 2

        return ((X - x0)**2 + (Y - y0)**2 <= R**2).astype(float)

    @staticmethod
    def _phase_psd(f, r0, L0, l0):
        f0 = 1 / L0
        fm = 5.92 / (2 * np.pi * l0)

        return (
            0.023
            * r0**(-5 / 3)
            * np.exp(-(f / fm)**2)
            / (f**2 + f0**2)**(11 / 6)
        )

    @classmethod
    def get_phase_screen(
        cls,
        N,
        dx,
        r0,
        L0=10.0,
        l0=0.001,
        n_subharmonics=3,
        seed=None,
        rng=None
    ):
        if rng is None:
            rng = np.random.default_rng(seed)

        D = N * dx
        df = 1 / D

        f = np.fft.fftfreq(N, d=dx)
        FX, FY = np.meshgrid(f, f, indexing="xy")
        F = np.hypot(FX, FY)

        PSD = cls._phase_psd(F, r0, L0, l0)
        PSD[0, 0] = 0

        noise = (
            rng.standard_normal((N, N))
            + 1j * rng.standard_normal((N, N))
        )

        spectrum = noise * np.sqrt(PSD) * df
        phase = np.fft.ifft2(spectrum).real * N**2

        if n_subharmonics > 0:
            x = (np.arange(N) - N // 2) * dx
            X, Y = np.meshgrid(x, x, indexing="xy")

            phase_low = np.zeros((N, N), dtype=float)

            for level in range(1, n_subharmonics + 1):
                df_sub = 1 / (3**level * D)
                frequencies = np.arange(-1, 2) * df_sub

                for fy in frequencies:
                    for fx in frequencies:
                        if fx == 0 and fy == 0:
                            continue

                        f_abs = np.hypot(fx, fy)
                        PSD_value = cls._phase_psd(
                            f_abs,
                            r0,
                            L0,
                            l0
                        )

                        coefficient = (
                            rng.standard_normal()
                            + 1j * rng.standard_normal()
                        )

                        coefficient *= np.sqrt(PSD_value) * df_sub

                        phase_low += np.real(
                            coefficient
                            * np.exp(
                                2j * np.pi * (fx * X + fy * Y)
                            )
                        )

            phase += phase_low

        return phase - phase.mean()

    @classmethod
    def propagate(
        cls,
        N,
        dx,
        E0,
        wavelength,
        distance,
        method="exact"
    ):
        E0 = np.asarray(E0, dtype=complex)

        f = np.fft.fftfreq(N, d=dx)
        FX, FY = np.meshgrid(f, f, indexing="xy")

        k = 2 * np.pi / wavelength

        if method == "paraxial":
            H = (
                np.exp(1j * k * distance)
                * np.exp(
                    -1j
                    * np.pi
                    * wavelength
                    * distance
                    * (FX**2 + FY**2)
                )
            )

        elif method == "exact":
            kx = 2 * np.pi * FX
            ky = 2 * np.pi * FY

            kz2 = k**2 - kx**2 - ky**2
            mask = kz2 >= 0

            H = np.zeros((N, N), dtype=complex)
            H[mask] = np.exp(
                1j * distance * np.sqrt(kz2[mask])
            )

        else:
            raise ValueError(
                "method должен быть 'exact' или 'paraxial'"
            )

        return np.fft.ifft2(np.fft.fft2(E0) * H)

    @classmethod
    def propagator_turbulent_atmosphere(
        cls,
        N,
        dx,
        E0,
        wavelength,
        distance,
        n_phase_screen,
        r0=0.1,
        L0=10.0,
        l0=0.001,
        method="exact",
        r0_is_total=True,
        n_subharmonics=3,
        seed=None
    ):
        rng = np.random.default_rng(seed)

        E = np.asarray(E0, dtype=complex).copy()
        dz = distance / n_phase_screen

        if r0_is_total:
            r0_screen = r0 * n_phase_screen**(3 / 5)
        else:
            r0_screen = r0

        E = cls.propagate(
            N,
            dx,
            E,
            wavelength,
            dz / 2,
            method
        )

        for i in range(n_phase_screen):
            phase = cls.get_phase_screen(
                N,
                dx,
                r0_screen,
                L0,
                l0,
                n_subharmonics=n_subharmonics,
                rng=rng
            )

            E *= np.exp(1j * phase)

            if i == n_phase_screen - 1:
                step = dz / 2
            else:
                step = dz

            E = cls.propagate(
                N,
                dx,
                E,
                wavelength,
                step,
                method
            )

        return E
    
class ShackHartmann:
    def __init__(
        self,
        size=1024,
        num_subapertures=4,
        period=0.011,
        radius_subapertures=0.005,
        focal_length=0.1,
        wavelength=1e-6,
        mode="square"
    ):
        self.size = size
        self.num_subapertures = num_subapertures
        self.period = period
        self.radius_subapertures = radius_subapertures
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.mode = mode

        self.screen_size = num_subapertures * period
        self.pixel_size = self.screen_size / size
        self.k = 2 * np.pi / wavelength

        self.X, self.Y = self._create_coordinate_grid()
        self.coordinate_centers = self._get_subaperture_centers()

        self.aperture_mask = self.create_aperture_mask()
        self.phase_mask = self.create_phase_mask()

    def _create_coordinate_grid(self):
        x = (np.arange(self.size) - self.size // 2) * self.pixel_size
        return np.meshgrid(x, x, indexing="xy")

    def _get_subaperture_centers(self):
        indices = np.arange(self.num_subapertures)
        indices = indices - (self.num_subapertures - 1) / 2

        centers = []

        for iy in indices:
            for ix in indices:
                x = ix * self.period
                y = iy * self.period

                if self.mode == "square":
                    centers.append((x, y))

                elif self.mode == "radial":
                    max_radius = self.screen_size / 2

                    if (
                        np.hypot(x, y) + self.radius_subapertures
                        <= max_radius
                    ):
                        centers.append((x, y))

                else:
                    raise ValueError(
                        "mode должен быть 'square' или 'radial'"
                    )

        return np.asarray(centers)

    def create_aperture_mask(self):
        mask = np.zeros(
            (self.size, self.size),
            dtype=float
        )

        for x0, y0 in self.coordinate_centers:
            local_mask = (
                (self.X - x0) ** 2
                + (self.Y - y0) ** 2
                <= self.radius_subapertures ** 2
            )

            mask[local_mask] = 1.0

        return mask

    def create_phase_mask(self):
        phase = np.zeros(
            (self.size, self.size),
            dtype=float
        )

        for x0, y0 in self.coordinate_centers:
            r2 = (
                (self.X - x0) ** 2
                + (self.Y - y0) ** 2
            )

            local_mask = (
                r2 <= self.radius_subapertures ** 2
            )

            phase[local_mask] = (
                -self.k
                * r2[local_mask]
                / (2 * self.focal_length)
            )

        return phase

    def apply(self, E):
        E = np.asarray(E, dtype=complex)

        if E.shape != (self.size, self.size):
            raise ValueError(
                f"E должно иметь форму "
                f"{(self.size, self.size)}"
            )

        return (
            E
            * self.aperture_mask
            * np.exp(1j * self.phase_mask)
        )

    def E_focal(self, E, method="exact"):
        E0 = self.apply(E)

        return Propagator.propagate(
            self.size,
            self.pixel_size,
            E0,
            self.wavelength,
            self.focal_length,
            method
        )

    def E_on_dist(self, E, alpha, method="exact"):
        E0 = self.apply(E)

        return Propagator.propagate(
            self.size,
            self.pixel_size,
            E0,
            self.wavelength,
            self.focal_length * alpha,
            method
        )

    def focal_intensity(self, E, method="exact"):
        return np.abs(
            self.E_focal(E, method)
        ) ** 2

    def _get_subaperture_slice(self, x0, y0):
        patch_size = int(
            round(self.period / self.pixel_size)
        )

        center_x = (
            int(round(x0 / self.pixel_size))
            + self.size // 2
        )

        center_y = (
            int(round(y0 / self.pixel_size))
            + self.size // 2
        )

        x_start = center_x - patch_size // 2
        y_start = center_y - patch_size // 2

        x_end = x_start + patch_size
        y_end = y_start + patch_size

        return (
            slice(y_start, y_end),
            slice(x_start, x_end)
        )

    def focal_spots(self, E, padding=4):
        E = np.asarray(E, dtype=complex)

        if E.shape != (self.size, self.size):
            raise ValueError(
                f"E должно иметь форму "
                f"{(self.size, self.size)}"
            )

        patch_size = int(
            round(self.period / self.pixel_size)
        )

        fft_size = padding * patch_size

        frequencies = np.fft.fftshift(
            np.fft.fftfreq(
                fft_size,
                d=self.pixel_size
            )
        )

        focal_coordinates = (
            self.wavelength
            * self.focal_length
            * frequencies
        )

        spots = []

        for x0, y0 in self.coordinate_centers:
            ys, xs = self._get_subaperture_slice(
                x0,
                y0
            )

            pupil = (
                E[ys, xs]
                * self.aperture_mask[ys, xs]
            )

            spectrum = np.fft.fftshift(
                np.fft.fft2(
                    pupil,
                    s=(fft_size, fft_size)
                )
            )

            intensity = np.abs(spectrum) ** 2
            spots.append(intensity)

        return (
            np.asarray(spots),
            focal_coordinates,
            focal_coordinates
        )

    def spot_centroids(
        self,
        E,
        padding=4,
        threshold=0.0,
        reference=None
    ):
        spots, x_focal, y_focal = self.focal_spots(
            E,
            padding
        )

        Xf, Yf = np.meshgrid(
            x_focal,
            y_focal,
            indexing="xy"
        )

        centroids = []

        for intensity in spots:
            intensity = intensity.copy()

            if threshold > 0:
                intensity[
                    intensity
                    < threshold * intensity.max()
                ] = 0

            norm = intensity.sum()

            if norm == 0:
                centroids.append((np.nan, np.nan))
                continue

            x_center = np.sum(
                Xf * intensity
            ) / norm

            y_center = np.sum(
                Yf * intensity
            ) / norm

            centroids.append(
                (x_center, y_center)
            )

        centroids = np.asarray(centroids)

        if reference is not None:
            centroids = centroids - np.asarray(reference)

        return centroids

    def reference_centroids(
        self,
        padding=4,
        threshold=0.0
    ):
        E_reference = np.ones(
            (self.size, self.size),
            dtype=complex
        )

        return self.spot_centroids(
            E_reference,
            padding=padding,
            threshold=threshold
        )

    def wavefront_slopes(
        self,
        E,
        padding=4,
        threshold=0.0,
        reference=None
    ):
        centroids = self.spot_centroids(
            E,
            padding=padding,
            threshold=threshold,
            reference=reference
        )

        return (
            self.k
            * centroids
            / self.focal_length
        )
    
class ApertureFunctions:
    WYANT_NM = {
        0: (0, 0),
        1: (1, 1),
        2: (1, -1),
        3: (2, 0),
        4: (2, 2),
        5: (2, -2),
        6: (3, 1),
        7: (3, -1),
        8: (4, 0),
        9: (3, 3),
        10: (3, -3),
        11: (4, 2),
        12: (4, -2),
        13: (5, 1),
        14: (5, -1),
        15: (6, 0),
        16: (4, 4),
        17: (4, -4),
        18: (5, 3),
        19: (5, -3),
        20: (6, 2),
        21: (6, -2),
        22: (7, 1),
        23: (7, -1),
        24: (8, 0),
        25: (5, 5),
        26: (5, -5),
        27: (6, 4),
        28: (6, -4),
        29: (7, 3),
        30: (7, -3),
        31: (8, 2),
        32: (8, -2),
        33: (9, 1),
        34: (9, -1),
        35: (10, 0),
        36: (6, 6),
        37: (6, -6),
        38: (7, 5),
        39: (7, -5),
        40: (8, 4),
        41: (8, -4),
        42: (9, 3),
        43: (9, -3),
        44: (10, 2),
        45: (10, -2),
        46: (11, 1),
        47: (11, -1),
        48: (12, 0),
        49: (7, 7),
        50: (7, -7),
        51: (8, 6),
        52: (8, -6),
        53: (9, 5),
        54: (9, -5),
        55: (10, 4),
        56: (10, -4),
        57: (11, 3),
        58: (11, -3),
        59: (12, 2),
        60: (12, -2),
        61: (13, 1),
        62: (13, -1),
        63: (14, 0),
        64: (8, 8),
        65: (8, -8),
        66: (9, 7),
        67: (9, -7),
        68: (10, 6),
        69: (10, -6),
        70: (11, 5),
        71: (11, -5),
        72: (12, 4),
        73: (12, -4),
        74: (13, 3),
        75: (13, -3),
        76: (14, 2),
        77: (14, -2),
        78: (15, 1),
        79: (15, -1),
        80: (16, 0),
        81: (9, 9),
        82: (9, -9),
        83: (10, 8),
        84: (10, -8),
        85: (11, 7),
        86: (11, -7),
        87: (12, 6),
        88: (12, -6),
        89: (13, 5),
        90: (13, -5),
        91: (14, 4),
        92: (14, -4),
        93: (15, 3),
        94: (15, -3),
        95: (16, 2),
        96: (16, -2),
        97: (17, 1),
        98: (17, -1),
        99: (18, 0),
        100: (10, 10),
        101: (10, -10),
        102: (11, 9),
        103: (11, -9),
        104: (12, 8),
        105: (12, -8),
        106: (13, 7),
        107: (13, -7),
        108: (14, 6),
        109: (14, -6),
        110: (15, 5),
        111: (15, -5),
        112: (16, 4),
        113: (16, -4),
        114: (17, 3),
        115: (17, -3),
        116: (18, 2),
        117: (18, -2),
        118: (19, 1),
        119: (19, -1),
        120: (20, 0),
    }

    ZERNIKE_FUNCTIONS = {
        0: lambda x, y: np.ones_like(x, dtype=float),
        1: lambda x, y: x,
        2: lambda x, y: y,
        3: lambda x, y: 2*(x*x + y*y) - 1,
        4: lambda x, y: (x - y)*(x + y),
        5: lambda x, y: 2*x*y,
        6: lambda x, y: (3*(x*x + y*y) - 2) * (x),
        7: lambda x, y: (3*(x*x + y*y) - 2) * (y),
        8: lambda x, y: 6*(x*x + y*y)**2 - 6*(x*x + y*y) + 1,
        9: lambda x, y: x*(x**2 - 3*y**2),
        10: lambda x, y: y*(3*x**2 - y**2),
        11: lambda x, y: (4*(x*x + y*y) - 3) * ((x - y)*(x + y)),
        12: lambda x, y: (4*(x*x + y*y) - 3) * (2*x*y),
        13: lambda x, y: (10*(x*x + y*y)**2 - 12*(x*x + y*y) + 3) * (x),
        14: lambda x, y: (10*(x*x + y*y)**2 - 12*(x*x + y*y) + 3) * (y),
        15: lambda x, y: (2*(x*x + y*y) - 1)*(10*(x*x + y*y)**2 - 10*(x*x + y*y) + 1),
        16: lambda x, y: (x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2),
        17: lambda x, y: 4*x*y*(x - y)*(x + y),
        18: lambda x, y: (5*(x*x + y*y) - 4) * (x*(x**2 - 3*y**2)),
        19: lambda x, y: (5*(x*x + y*y) - 4) * (y*(3*x**2 - y**2)),
        20: lambda x, y: (15*(x*x + y*y)**2 - 20*(x*x + y*y) + 6) * ((x - y)*(x + y)),
        21: lambda x, y: (15*(x*x + y*y)**2 - 20*(x*x + y*y) + 6) * (2*x*y),
        22: lambda x, y: (35*(x*x + y*y)**3 - 60*(x*x + y*y)**2 + 30*(x*x + y*y) - 4) * (x),
        23: lambda x, y: (35*(x*x + y*y)**3 - 60*(x*x + y*y)**2 + 30*(x*x + y*y) - 4) * (y),
        24: lambda x, y: 70*(x*x + y*y)**4 - 140*(x*x + y*y)**3 + 90*(x*x + y*y)**2 - 20*(x*x + y*y) + 1,
        25: lambda x, y: x*(x**4 - 10*x**2*y**2 + 5*y**4),
        26: lambda x, y: y*(5*x**4 - 10*x**2*y**2 + y**4),
        27: lambda x, y: (6*(x*x + y*y) - 5) * ((x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        28: lambda x, y: (6*(x*x + y*y) - 5) * (4*x*y*(x - y)*(x + y)),
        29: lambda x, y: (21*(x*x + y*y)**2 - 30*(x*x + y*y) + 10) * (x*(x**2 - 3*y**2)),
        30: lambda x, y: (21*(x*x + y*y)**2 - 30*(x*x + y*y) + 10) * (y*(3*x**2 - y**2)),
        31: lambda x, y: (56*(x*x + y*y)**3 - 105*(x*x + y*y)**2 + 60*(x*x + y*y) - 10) * ((x - y)*(x + y)),
        32: lambda x, y: (56*(x*x + y*y)**3 - 105*(x*x + y*y)**2 + 60*(x*x + y*y) - 10) * (2*x*y),
        33: lambda x, y: (126*(x*x + y*y)**4 - 280*(x*x + y*y)**3 + 210*(x*x + y*y)**2 - 60*(x*x + y*y) + 5) * (x),
        34: lambda x, y: (126*(x*x + y*y)**4 - 280*(x*x + y*y)**3 + 210*(x*x + y*y)**2 - 60*(x*x + y*y) + 5) * (y),
        35: lambda x, y: (2*(x*x + y*y) - 1)*(126*(x*x + y*y)**4 - 252*(x*x + y*y)**3 + 154*(x*x + y*y)**2 - 28*(x*x + y*y) + 1),
        36: lambda x, y: (x - y)*(x + y)*(x**2 - 4*x*y + y**2)*(x**2 + 4*x*y + y**2),
        37: lambda x, y: 2*x*y*(x**2 - 3*y**2)*(3*x**2 - y**2),
        38: lambda x, y: (7*(x*x + y*y) - 6) * (x*(x**4 - 10*x**2*y**2 + 5*y**4)),
        39: lambda x, y: (7*(x*x + y*y) - 6) * (y*(5*x**4 - 10*x**2*y**2 + y**4)),
        40: lambda x, y: (28*(x*x + y*y)**2 - 42*(x*x + y*y) + 15) * ((x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        41: lambda x, y: (28*(x*x + y*y)**2 - 42*(x*x + y*y) + 15) * (4*x*y*(x - y)*(x + y)),
        42: lambda x, y: (84*(x*x + y*y)**3 - 168*(x*x + y*y)**2 + 105*(x*x + y*y) - 20) * (x*(x**2 - 3*y**2)),
        43: lambda x, y: (84*(x*x + y*y)**3 - 168*(x*x + y*y)**2 + 105*(x*x + y*y) - 20) * (y*(3*x**2 - y**2)),
        44: lambda x, y: (210*(x*x + y*y)**4 - 504*(x*x + y*y)**3 + 420*(x*x + y*y)**2 - 140*(x*x + y*y) + 15) * ((x - y)*(x + y)),
        45: lambda x, y: (210*(x*x + y*y)**4 - 504*(x*x + y*y)**3 + 420*(x*x + y*y)**2 - 140*(x*x + y*y) + 15) * (2*x*y),
        46: lambda x, y: (462*(x*x + y*y)**5 - 1260*(x*x + y*y)**4 + 1260*(x*x + y*y)**3 - 560*(x*x + y*y)**2 + 105*(x*x + y*y) - 6) * (x),
        47: lambda x, y: (462*(x*x + y*y)**5 - 1260*(x*x + y*y)**4 + 1260*(x*x + y*y)**3 - 560*(x*x + y*y)**2 + 105*(x*x + y*y) - 6) * (y),
        48: lambda x, y: 924*(x*x + y*y)**6 - 2772*(x*x + y*y)**5 + 3150*(x*x + y*y)**4 - 1680*(x*x + y*y)**3 + 420*(x*x + y*y)**2 - 42*(x*x + y*y) + 1,
        49: lambda x, y: x*(x**6 - 21*x**4*y**2 + 35*x**2*y**4 - 7*y**6),
        50: lambda x, y: y*(7*x**6 - 35*x**4*y**2 + 21*x**2*y**4 - y**6),
        51: lambda x, y: (8*(x*x + y*y) - 7) * ((x - y)*(x + y)*(x**2 - 4*x*y + y**2)*(x**2 + 4*x*y + y**2)),
        52: lambda x, y: (8*(x*x + y*y) - 7) * (2*x*y*(x**2 - 3*y**2)*(3*x**2 - y**2)),
        53: lambda x, y: (36*(x*x + y*y)**2 - 56*(x*x + y*y) + 21) * (x*(x**4 - 10*x**2*y**2 + 5*y**4)),
        54: lambda x, y: (36*(x*x + y*y)**2 - 56*(x*x + y*y) + 21) * (y*(5*x**4 - 10*x**2*y**2 + y**4)),
        55: lambda x, y: (120*(x*x + y*y)**3 - 252*(x*x + y*y)**2 + 168*(x*x + y*y) - 35) * ((x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        56: lambda x, y: (120*(x*x + y*y)**3 - 252*(x*x + y*y)**2 + 168*(x*x + y*y) - 35) * (4*x*y*(x - y)*(x + y)),
        57: lambda x, y: (330*(x*x + y*y)**4 - 840*(x*x + y*y)**3 + 756*(x*x + y*y)**2 - 280*(x*x + y*y) + 35) * (x*(x**2 - 3*y**2)),
        58: lambda x, y: (330*(x*x + y*y)**4 - 840*(x*x + y*y)**3 + 756*(x*x + y*y)**2 - 280*(x*x + y*y) + 35) * (y*(3*x**2 - y**2)),
        59: lambda x, y: (792*(x*x + y*y)**5 - 2310*(x*x + y*y)**4 + 2520*(x*x + y*y)**3 - 1260*(x*x + y*y)**2 + 280*(x*x + y*y) - 21) * ((x - y)*(x + y)),
        60: lambda x, y: (792*(x*x + y*y)**5 - 2310*(x*x + y*y)**4 + 2520*(x*x + y*y)**3 - 1260*(x*x + y*y)**2 + 280*(x*x + y*y) - 21) * (2*x*y),
        61: lambda x, y: (1716*(x*x + y*y)**6 - 5544*(x*x + y*y)**5 + 6930*(x*x + y*y)**4 - 4200*(x*x + y*y)**3 + 1260*(x*x + y*y)**2 - 168*(x*x + y*y) + 7) * (x),
        62: lambda x, y: (1716*(x*x + y*y)**6 - 5544*(x*x + y*y)**5 + 6930*(x*x + y*y)**4 - 4200*(x*x + y*y)**3 + 1260*(x*x + y*y)**2 - 168*(x*x + y*y) + 7) * (y),
        63: lambda x, y: (2*(x*x + y*y) - 1)*(1716*(x*x + y*y)**6 - 5148*(x*x + y*y)**5 + 5742*(x*x + y*y)**4 - 2904*(x*x + y*y)**3 + 648*(x*x + y*y)**2 - 54*(x*x + y*y) + 1),
        64: lambda x, y: (x**4 - 4*x**3*y - 6*x**2*y**2 + 4*x*y**3 + y**4)*(x**4 + 4*x**3*y - 6*x**2*y**2 - 4*x*y**3 + y**4),
        65: lambda x, y: 8*x*y*(x - y)*(x + y)*(x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2),
        66: lambda x, y: (9*(x*x + y*y) - 8) * (x*(x**6 - 21*x**4*y**2 + 35*x**2*y**4 - 7*y**6)),
        67: lambda x, y: (9*(x*x + y*y) - 8) * (y*(7*x**6 - 35*x**4*y**2 + 21*x**2*y**4 - y**6)),
        68: lambda x, y: ((3*(x*x + y*y) - 2)*(15*(x*x + y*y) - 14)) * ((x - y)*(x + y)*(x**2 - 4*x*y + y**2)*(x**2 + 4*x*y + y**2)),
        69: lambda x, y: ((3*(x*x + y*y) - 2)*(15*(x*x + y*y) - 14)) * (2*x*y*(x**2 - 3*y**2)*(3*x**2 - y**2)),
        70: lambda x, y: (165*(x*x + y*y)**3 - 360*(x*x + y*y)**2 + 252*(x*x + y*y) - 56) * (x*(x**4 - 10*x**2*y**2 + 5*y**4)),
        71: lambda x, y: (165*(x*x + y*y)**3 - 360*(x*x + y*y)**2 + 252*(x*x + y*y) - 56) * (y*(5*x**4 - 10*x**2*y**2 + y**4)),
        72: lambda x, y: (495*(x*x + y*y)**4 - 1320*(x*x + y*y)**3 + 1260*(x*x + y*y)**2 - 504*(x*x + y*y) + 70) * ((x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        73: lambda x, y: (495*(x*x + y*y)**4 - 1320*(x*x + y*y)**3 + 1260*(x*x + y*y)**2 - 504*(x*x + y*y) + 70) * (4*x*y*(x - y)*(x + y)),
        74: lambda x, y: (1287*(x*x + y*y)**5 - 3960*(x*x + y*y)**4 + 4620*(x*x + y*y)**3 - 2520*(x*x + y*y)**2 + 630*(x*x + y*y) - 56) * (x*(x**2 - 3*y**2)),
        75: lambda x, y: (1287*(x*x + y*y)**5 - 3960*(x*x + y*y)**4 + 4620*(x*x + y*y)**3 - 2520*(x*x + y*y)**2 + 630*(x*x + y*y) - 56) * (y*(3*x**2 - y**2)),
        76: lambda x, y: (3003*(x*x + y*y)**6 - 10296*(x*x + y*y)**5 + 13860*(x*x + y*y)**4 - 9240*(x*x + y*y)**3 + 3150*(x*x + y*y)**2 - 504*(x*x + y*y) + 28) * ((x - y)*(x + y)),
        77: lambda x, y: (3003*(x*x + y*y)**6 - 10296*(x*x + y*y)**5 + 13860*(x*x + y*y)**4 - 9240*(x*x + y*y)**3 + 3150*(x*x + y*y)**2 - 504*(x*x + y*y) + 28) * (2*x*y),
        78: lambda x, y: (6435*(x*x + y*y)**7 - 24024*(x*x + y*y)**6 + 36036*(x*x + y*y)**5 - 27720*(x*x + y*y)**4 + 11550*(x*x + y*y)**3 - 2520*(x*x + y*y)**2 + 252*(x*x + y*y) - 8) * (x),
        79: lambda x, y: (6435*(x*x + y*y)**7 - 24024*(x*x + y*y)**6 + 36036*(x*x + y*y)**5 - 27720*(x*x + y*y)**4 + 11550*(x*x + y*y)**3 - 2520*(x*x + y*y)**2 + 252*(x*x + y*y) - 8) * (y),
        80: lambda x, y: 12870*(x*x + y*y)**8 - 51480*(x*x + y*y)**7 + 84084*(x*x + y*y)**6 - 72072*(x*x + y*y)**5 + 34650*(x*x + y*y)**4 - 9240*(x*x + y*y)**3 + 1260*(x*x + y*y)**2 - 72*(x*x + y*y) + 1,
        81: lambda x, y: x*(x**2 - 3*y**2)*(x**6 - 33*x**4*y**2 + 27*x**2*y**4 - 3*y**6),
        82: lambda x, y: y*(3*x**2 - y**2)*(3*x**6 - 27*x**4*y**2 + 33*x**2*y**4 - y**6),
        83: lambda x, y: (10*(x*x + y*y) - 9) * ((x**4 - 4*x**3*y - 6*x**2*y**2 + 4*x*y**3 + y**4)*(x**4 + 4*x**3*y - 6*x**2*y**2 - 4*x*y**3 + y**4)),
        84: lambda x, y: (10*(x*x + y*y) - 9) * (8*x*y*(x - y)*(x + y)*(x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        85: lambda x, y: (55*(x*x + y*y)**2 - 90*(x*x + y*y) + 36) * (x*(x**6 - 21*x**4*y**2 + 35*x**2*y**4 - 7*y**6)),
        86: lambda x, y: (55*(x*x + y*y)**2 - 90*(x*x + y*y) + 36) * (y*(7*x**6 - 35*x**4*y**2 + 21*x**2*y**4 - y**6)),
        87: lambda x, y: (220*(x*x + y*y)**3 - 495*(x*x + y*y)**2 + 360*(x*x + y*y) - 84) * ((x - y)*(x + y)*(x**2 - 4*x*y + y**2)*(x**2 + 4*x*y + y**2)),
        88: lambda x, y: (220*(x*x + y*y)**3 - 495*(x*x + y*y)**2 + 360*(x*x + y*y) - 84) * (2*x*y*(x**2 - 3*y**2)*(3*x**2 - y**2)),
        89: lambda x, y: (715*(x*x + y*y)**4 - 1980*(x*x + y*y)**3 + 1980*(x*x + y*y)**2 - 840*(x*x + y*y) + 126) * (x*(x**4 - 10*x**2*y**2 + 5*y**4)),
        90: lambda x, y: (715*(x*x + y*y)**4 - 1980*(x*x + y*y)**3 + 1980*(x*x + y*y)**2 - 840*(x*x + y*y) + 126) * (y*(5*x**4 - 10*x**2*y**2 + y**4)),
        91: lambda x, y: (2002*(x*x + y*y)**5 - 6435*(x*x + y*y)**4 + 7920*(x*x + y*y)**3 - 4620*(x*x + y*y)**2 + 1260*(x*x + y*y) - 126) * ((x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        92: lambda x, y: (2002*(x*x + y*y)**5 - 6435*(x*x + y*y)**4 + 7920*(x*x + y*y)**3 - 4620*(x*x + y*y)**2 + 1260*(x*x + y*y) - 126) * (4*x*y*(x - y)*(x + y)),
        93: lambda x, y: (5005*(x*x + y*y)**6 - 18018*(x*x + y*y)**5 + 25740*(x*x + y*y)**4 - 18480*(x*x + y*y)**3 + 6930*(x*x + y*y)**2 - 1260*(x*x + y*y) + 84) * (x*(x**2 - 3*y**2)),
        94: lambda x, y: (5005*(x*x + y*y)**6 - 18018*(x*x + y*y)**5 + 25740*(x*x + y*y)**4 - 18480*(x*x + y*y)**3 + 6930*(x*x + y*y)**2 - 1260*(x*x + y*y) + 84) * (y*(3*x**2 - y**2)),
        95: lambda x, y: (11440*(x*x + y*y)**7 - 45045*(x*x + y*y)**6 + 72072*(x*x + y*y)**5 - 60060*(x*x + y*y)**4 + 27720*(x*x + y*y)**3 - 6930*(x*x + y*y)**2 + 840*(x*x + y*y) - 36) * ((x - y)*(x + y)),
        96: lambda x, y: (11440*(x*x + y*y)**7 - 45045*(x*x + y*y)**6 + 72072*(x*x + y*y)**5 - 60060*(x*x + y*y)**4 + 27720*(x*x + y*y)**3 - 6930*(x*x + y*y)**2 + 840*(x*x + y*y) - 36) * (2*x*y),
        97: lambda x, y: (24310*(x*x + y*y)**8 - 102960*(x*x + y*y)**7 + 180180*(x*x + y*y)**6 - 168168*(x*x + y*y)**5 + 90090*(x*x + y*y)**4 - 27720*(x*x + y*y)**3 + 4620*(x*x + y*y)**2 - 360*(x*x + y*y) + 9) * (x),
        98: lambda x, y: (24310*(x*x + y*y)**8 - 102960*(x*x + y*y)**7 + 180180*(x*x + y*y)**6 - 168168*(x*x + y*y)**5 + 90090*(x*x + y*y)**4 - 27720*(x*x + y*y)**3 + 4620*(x*x + y*y)**2 - 360*(x*x + y*y) + 9) * (y),
        99: lambda x, y: (2*(x*x + y*y) - 1)*(24310*(x*x + y*y)**8 - 97240*(x*x + y*y)**7 + 157300*(x*x + y*y)**6 - 131560*(x*x + y*y)**5 + 60346*(x*x + y*y)**4 - 14872*(x*x + y*y)**3 + 1804*(x*x + y*y)**2 - 88*(x*x + y*y) + 1),
        100: lambda x, y: (x - y)*(x + y)*(x**4 - 4*x**3*y - 14*x**2*y**2 - 4*x*y**3 + y**4)*(x**4 + 4*x**3*y - 14*x**2*y**2 + 4*x*y**3 + y**4),
        101: lambda x, y: 2*x*y*(x**4 - 10*x**2*y**2 + 5*y**4)*(5*x**4 - 10*x**2*y**2 + y**4),
        102: lambda x, y: (11*(x*x + y*y) - 10) * (x*(x**2 - 3*y**2)*(x**6 - 33*x**4*y**2 + 27*x**2*y**4 - 3*y**6)),
        103: lambda x, y: (11*(x*x + y*y) - 10) * (y*(3*x**2 - y**2)*(3*x**6 - 27*x**4*y**2 + 33*x**2*y**4 - y**6)),
        104: lambda x, y: (66*(x*x + y*y)**2 - 110*(x*x + y*y) + 45) * ((x**4 - 4*x**3*y - 6*x**2*y**2 + 4*x*y**3 + y**4)*(x**4 + 4*x**3*y - 6*x**2*y**2 - 4*x*y**3 + y**4)),
        105: lambda x, y: (66*(x*x + y*y)**2 - 110*(x*x + y*y) + 45) * (8*x*y*(x - y)*(x + y)*(x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        106: lambda x, y: (286*(x*x + y*y)**3 - 660*(x*x + y*y)**2 + 495*(x*x + y*y) - 120) * (x*(x**6 - 21*x**4*y**2 + 35*x**2*y**4 - 7*y**6)),
        107: lambda x, y: (286*(x*x + y*y)**3 - 660*(x*x + y*y)**2 + 495*(x*x + y*y) - 120) * (y*(7*x**6 - 35*x**4*y**2 + 21*x**2*y**4 - y**6)),
        108: lambda x, y: (1001*(x*x + y*y)**4 - 2860*(x*x + y*y)**3 + 2970*(x*x + y*y)**2 - 1320*(x*x + y*y) + 210) * ((x - y)*(x + y)*(x**2 - 4*x*y + y**2)*(x**2 + 4*x*y + y**2)),
        109: lambda x, y: (1001*(x*x + y*y)**4 - 2860*(x*x + y*y)**3 + 2970*(x*x + y*y)**2 - 1320*(x*x + y*y) + 210) * (2*x*y*(x**2 - 3*y**2)*(3*x**2 - y**2)),
        110: lambda x, y: (3003*(x*x + y*y)**5 - 10010*(x*x + y*y)**4 + 12870*(x*x + y*y)**3 - 7920*(x*x + y*y)**2 + 2310*(x*x + y*y) - 252) * (x*(x**4 - 10*x**2*y**2 + 5*y**4)),
        111: lambda x, y: (3003*(x*x + y*y)**5 - 10010*(x*x + y*y)**4 + 12870*(x*x + y*y)**3 - 7920*(x*x + y*y)**2 + 2310*(x*x + y*y) - 252) * (y*(5*x**4 - 10*x**2*y**2 + y**4)),
        112: lambda x, y: (8008*(x*x + y*y)**6 - 30030*(x*x + y*y)**5 + 45045*(x*x + y*y)**4 - 34320*(x*x + y*y)**3 + 13860*(x*x + y*y)**2 - 2772*(x*x + y*y) + 210) * ((x**2 - 2*x*y - y**2)*(x**2 + 2*x*y - y**2)),
        113: lambda x, y: (8008*(x*x + y*y)**6 - 30030*(x*x + y*y)**5 + 45045*(x*x + y*y)**4 - 34320*(x*x + y*y)**3 + 13860*(x*x + y*y)**2 - 2772*(x*x + y*y) + 210) * (4*x*y*(x - y)*(x + y)),
        114: lambda x, y: (19448*(x*x + y*y)**7 - 80080*(x*x + y*y)**6 + 135135*(x*x + y*y)**5 - 120120*(x*x + y*y)**4 + 60060*(x*x + y*y)**3 - 16632*(x*x + y*y)**2 + 2310*(x*x + y*y) - 120) * (x*(x**2 - 3*y**2)),
        115: lambda x, y: (19448*(x*x + y*y)**7 - 80080*(x*x + y*y)**6 + 135135*(x*x + y*y)**5 - 120120*(x*x + y*y)**4 + 60060*(x*x + y*y)**3 - 16632*(x*x + y*y)**2 + 2310*(x*x + y*y) - 120) * (y*(3*x**2 - y**2)),
        116: lambda x, y: (43758*(x*x + y*y)**8 - 194480*(x*x + y*y)**7 + 360360*(x*x + y*y)**6 - 360360*(x*x + y*y)**5 + 210210*(x*x + y*y)**4 - 72072*(x*x + y*y)**3 + 13860*(x*x + y*y)**2 - 1320*(x*x + y*y) + 45) * ((x - y)*(x + y)),
        117: lambda x, y: (43758*(x*x + y*y)**8 - 194480*(x*x + y*y)**7 + 360360*(x*x + y*y)**6 - 360360*(x*x + y*y)**5 + 210210*(x*x + y*y)**4 - 72072*(x*x + y*y)**3 + 13860*(x*x + y*y)**2 - 1320*(x*x + y*y) + 45) * (2*x*y),
        118: lambda x, y: (92378*(x*x + y*y)**9 - 437580*(x*x + y*y)**8 + 875160*(x*x + y*y)**7 - 960960*(x*x + y*y)**6 + 630630*(x*x + y*y)**5 - 252252*(x*x + y*y)**4 + 60060*(x*x + y*y)**3 - 7920*(x*x + y*y)**2 + 495*(x*x + y*y) - 10) * (x),
        119: lambda x, y: (92378*(x*x + y*y)**9 - 437580*(x*x + y*y)**8 + 875160*(x*x + y*y)**7 - 960960*(x*x + y*y)**6 + 630630*(x*x + y*y)**5 - 252252*(x*x + y*y)**4 + 60060*(x*x + y*y)**3 - 7920*(x*x + y*y)**2 + 495*(x*x + y*y) - 10) * (y),
        120: lambda x, y: 184756*(x*x + y*y)**10 - 923780*(x*x + y*y)**9 + 1969110*(x*x + y*y)**8 - 2333760*(x*x + y*y)**7 + 1681680*(x*x + y*y)**6 - 756756*(x*x + y*y)**5 + 210210*(x*x + y*y)**4 - 34320*(x*x + y*y)**3 + 2970*(x*x + y*y)**2 - 110*(x*x + y*y) + 1,
    }


    @classmethod
    def zernike_j(cls, j, X, Y):
        if j not in cls.ZERNIKE_FUNCTIONS:
            raise ValueError(f"Индекс Wyant должен лежать от 0 до {len(cls.ZERNIKE_FUNCTIONS) - 1}")
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.shape != Y.shape:
            raise ValueError("X и Y должны иметь одинаковую форму")
        return cls.ZERNIKE_FUNCTIONS[j](X, Y)

    @classmethod
    def zernike(cls, coeffs, X, Y):
        coeffs = np.asarray(coeffs)
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape != Y.shape:
            raise ValueError("X и Y должны иметь одинаковую форму")

        if coeffs.ndim != 1:
            raise ValueError("coeffs должен быть одномерным массивом")

        if len(coeffs) > len(cls.ZERNIKE_FUNCTIONS):
            raise ValueError(f"Поддерживается не более {len(cls.ZERNIKE_FUNCTIONS)} коэффициентов")

        dtype = np.result_type(coeffs.dtype, np.float64)
        result = np.zeros(X.shape, dtype=dtype)

        for j, coeff in enumerate(coeffs):
            if coeff != 0:
                result += coeff * cls.ZERNIKE_FUNCTIONS[j](X, Y)

        return np.where(X * X + Y * Y <= 1.0, result, 0.0)

    @classmethod
    def zernike_image(cls, coeffs, N):
        coordinates = np.linspace(-1.0, 1.0, N)
        X, Y = np.meshgrid(coordinates, coordinates, indexing="xy")
        return cls.zernike(coeffs, X, Y)

    @classmethod
    def reconstruct_zernike(cls, phase_measured, X_norm, Y_norm, num_coeffs=15):
        phase_measured = np.asarray(phase_measured)
        X_norm = np.asarray(X_norm)
        Y_norm = np.asarray(Y_norm)

        if phase_measured.shape != X_norm.shape or phase_measured.shape != Y_norm.shape:
            raise ValueError("phase_measured, X_norm и Y_norm должны иметь одинаковую форму")

        if not 1 <= num_coeffs <= len(cls.ZERNIKE_FUNCTIONS):
            raise ValueError(f"num_coeffs должен лежать от 1 до {len(cls.ZERNIKE_FUNCTIONS)}")

        mask = (
            (X_norm * X_norm + Y_norm * Y_norm <= 1.0)
            & np.isfinite(phase_measured)
            & np.isfinite(X_norm)
            & np.isfinite(Y_norm)
        )

        x = X_norm[mask]
        y = Y_norm[mask]
        phase = phase_measured[mask]

        coeffs = np.empty(num_coeffs, dtype=float)

        for j in range(num_coeffs):
            Z = cls.ZERNIKE_FUNCTIONS[j](x, y)
            coeffs[j] = np.dot(phase, Z) / np.dot(Z, Z)

        return coeffs
