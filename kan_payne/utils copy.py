import numpy as np


def load_model_params_from_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    weights = data["weights"]
    spline_weights = data["spline_weights"]
    spline_scalers = data["spline_scalers"]
    grids = data["grids"]
    spline_orders = data["spline_orders"]

    return weights, spline_weights, spline_scalers, grids, spline_orders


def doppler_shift(wave, flux, dv, wave_obs):
    """
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    """
    c = 2.99792458e5  # km/s
    doppler_factor = np.sqrt((1 - dv / c) / (1 + dv / c))
    new_wavelength = wave_obs * doppler_factor
    new_flux = np.interp(new_wavelength, wave, flux)
    return new_flux

