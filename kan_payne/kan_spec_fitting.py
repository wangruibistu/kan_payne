import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage

from . import kan_spec_model
from . import utils


def fit_normalized_spectrum_single_star_model_curvefit(
    spec_flux, spec_err, spec_wave, wave_model, KAN_coeffs, mask, x_min, x_max, p0=None
):

    tol = 5e-10  # tolerance for when the optimizer should stop optimizing.

    # set infinity uncertainty to pixels that we want to omit
    spec_err[mask] = 999.0

    # assuming a neural net that has two hidden layers.
    (
        kan_weights,
        kan_spline_weights,
        kan_spline_scalers,
        kan_grids,
        kan_spline_orders,
    ) = KAN_coeffs

    # number of labels + radial velocity
    num_labels = kan_weights[0].shape[-1] + 1

    # print("num_labels", num_labels)
    def fit_func(dummy_variable, *labels):
        norm_spec = kan_spec_model.get_spectrum_from_kan(
            scaled_labels=np.array([labels[:-1]]),
            KAN_coeffs=KAN_coeffs,
        )
        norm_spec = ndimage.gaussian_filter1d(norm_spec, sigma=1.97)
        norm_spec = utils.doppler_shift(wave_model, norm_spec, labels[-1], spec_wave)
        return norm_spec

    # if no initial guess is supplied, initialize with the median value
    if p0 is None:
        p0 = np.zeros(num_labels) + 0.5
        p0[-1] = 0
    # prohibit the minimimizer to go outside the range of training set
    bounds = np.zeros((2, num_labels))
    bounds[0, :] = 0
    bounds[1, :] = 1
    bounds[0, -1] = -500.0
    bounds[1, -1] = 500.0

    # run the optimizer
    popt, pcov = curve_fit(
        fit_func,
        xdata=[],
        ydata=spec_flux,
        sigma=spec_err,
        p0=p0,
        bounds=bounds,
        ftol=tol,
        xtol=tol,
        absolute_sigma=True,
        method="trf",
    )
    model_spec = fit_func([], *popt)
    pstd = np.sqrt(np.diagonal(pcov))
    # rescale the result back to original unit
    popt[:-1] = popt[:-1] * (x_max - x_min) + x_min
    pstd[:-1] = pstd[:-1] * (x_max - x_min)

    return popt, pstd, model_spec


from scipy.optimize import minimize


def fit_normalized_spectrum_single_star_model(
    spec_flux,
    spec_err,
    spec_wave,
    wave_model,
    KAN_coeffs,
    x_min,
    x_max,
    mask=None,
    p0=None,
):

    tol = 5e-10  # tolerance for when the optimizer should stop optimizing.

    # set infinity uncertainty to pixels that we want to omit
    spec_err[mask] = 999.0

    # assuming a neural net that has two hidden layers.

    # number of labels + radial velocity
    num_labels = 4

    def fit_func(labels):
        # print(labels)
        # print(np.array([labels[:-1]]).shape)
        norm_spec = kan_spec_model.get_spectrum_from_kan(
            scaled_labels=np.array([labels[:-1]]).reshape(1, 3),
            KAN_coeffs=KAN_coeffs,
        )
        norm_spec = ndimage.gaussian_filter1d(norm_spec, sigma=2.97)  # 1.97
        norm_spec = utils.doppler_shift(wave_model, norm_spec, labels[-1], spec_wave)
        return np.sum((norm_spec - spec_flux) ** 2 / spec_err**2)

    def produce_spec(labels):
        # print(labels)
        # print(np.array([labels[:-1]]).shape)
        norm_spec = kan_spec_model.get_spectrum_from_kan(
            scaled_labels=np.array([labels[:-1]]).reshape(1, 3),
            KAN_coeffs=KAN_coeffs,
        )
        norm_spec = ndimage.gaussian_filter1d(norm_spec, sigma=1.97)
        norm_spec = utils.doppler_shift(wave_model, norm_spec, labels[-1], spec_wave)
        return norm_spec

    # if no initial guess is supplied, initialize with the median value
    if p0 is None:
        p0 = np.zeros(num_labels) + 0.5
        p0[-1] = 0
    # prohibit the minimimizer to go outside the range of training set
    bounds = [(0, 1)] * (num_labels - 1) + [(-500, 500)]

    # run the optimizer
    result = minimize(
        fit_func,
        x0=p0,
        bounds=bounds,
        tol=tol,
        method="L-BFGS-B",
        options={"ftol": tol, "gtol": tol},
    )

    popt = result.x
    model_spec = produce_spec(popt)
    # 使用有限差分法估计协方差矩阵
    pcov = np.linalg.inv(result.hess_inv.todense())
    pstd = np.sqrt(np.diagonal(pcov))

    # rescale the result back to original unit
    popt[:-1] = popt[:-1] * (x_max - x_min) + x_min
    pstd[:-1] = pstd[:-1] * (x_max - x_min)
    # print(model_spec)
    return popt, pstd, model_spec
