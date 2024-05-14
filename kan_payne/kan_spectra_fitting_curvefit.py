# code for fitting spectra, using the models in spectral_model.py
import numpy as np
from scipy.optimize import curve_fit
import torch
import matplotlib.pyplot as plt
import os, sys
import sys

sys.path.append("/home/wangr/code/efficient-kan/src/")
from efficient_kan import KAN
from The_Payne import training, utils


dtype = torch.float
device = torch.device('cuda')
def fit_spectrum_single_star_kan_model(
    spec, spec_err, wavelength, mask, 
    kan_model, x_min, x_max, y_min, y_max, device, p0=None
):
    """
    fit a single-star model to a single combined spectrum

    p0 is an initial guess for where to initialize the optimizer. Because
        this is a simple model, having a good initial guess is usually not
        important.

    labels = [
        Teff, Logg, Vturb, \
        [C/H], [N/H], [O/H], [Na/H], [Mg/H], \
        [Al/H], [Si/H], [P/H], [S/H], [K/H], \
        [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H], \
        [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H], \
        C12/C13, Vmacro, rv
        ]

    returns:
        popt: the best-fit labels
        pstd: the formal fitting uncertainties
        model_spec: the model spectrum corresponding to popt
    """

    tol = 5e-4  # tolerance for when the optimizer should stop optimizing.

    # set infinity uncertainty to pixels that we want to omit
    spec_err[mask] = 9999.0

    # assuming a neural net that has two hidden layers.
    # w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = (
    #     NN_coeffs
    # )

    # number of labels + radial velocity
    # num_labels = w_array_0.shape[-1] + 1
    num_labels = 25 + 1

    def fit_func(dummy_variable, *labels):
        norm_spec = (
            kan_model(
                torch.tensor(labels[:-1]).view(1, 25).type(dtype).to(device),
            ).cpu().data.numpy()[0] * (y_max - y_min) + y_min
        )
        norm_spec = utils.doppler_shift(wavelength, norm_spec, labels[-1])
        return norm_spec

    # if no initial guess is supplied, initialize with the median value
    if p0 is None:
        p0 = np.zeros(num_labels) + 0.5

    # prohibit the minimimizer to go outside the range of training set
    bounds = np.zeros((2, num_labels))
    bounds[0, :] = 0
    bounds[1, :] = 1
    bounds[0, -1] = -5.0
    bounds[1, -1] = 5.0

    # run the optimizer
    popt, pcov = curve_fit(
        fit_func,
        xdata=[],
        ydata=spec,
        sigma=spec_err,
        p0=p0,
        bounds=bounds,
        ftol=tol,
        xtol=tol,
        absolute_sigma=False,
        method="trf",
    )
    pstd = np.sqrt(np.diag(pcov))
    model_spec = fit_func([], *popt)

    # rescale the result back to original unit
    popt[:-1] = popt[:-1] * (x_max - x_min) + x_min
    pstd[:-1] = pstd[:-1] * (x_max - x_min)
    return popt, pstd, model_spec


if __name__ == "__main__":
    training_labels, training_spectra, validation_labels, validation_spectra = (
        utils.load_training_data()
    )

    x_max = np.max(training_labels, axis=0)
    x_min = np.min(training_labels, axis=0)

    y_max = np.max(training_spectra, axis=0)
    y_min = np.min(training_spectra, axis=0)

    model = torch.load("./model_save/Payne_KAN_model_01.kpt")
    model.to(device)

    wave = np.load(
        "/home/wangr/code/The_Payne_KAN/The_Payne_KAN/other_data/apogee_wavelength.npz"
    )
    wavelength = wave["wavelength"]

    spec_err = 1e-2 * np.ones(len(wavelength))

    real_labels = scaled_labels = [
        5777,
        4.44,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        90.0,
        6.0,
        3.0,
    ]  

    scaled_labels[:-1] = (real_labels[:-1] - x_min) / (x_max - x_min)  
    print(np.array(scaled_labels).shape)

    flux_pred = model(
        torch.tensor(scaled_labels[:-1]).view(1, 25).type(dtype).to(device=device)
        ).cpu().data.numpy()[0]
    real_spec = flux_pred * (y_max - y_min) + y_min
    spec = utils.doppler_shift(wavelength, real_spec, scaled_labels[-1])


    mask = np.zeros(len(wavelength), dtype=bool)  

    popt, pstd, model_spec = fit_spectrum_single_star_kan_model(
        spec=spec,
        spec_err=spec_err,
        wavelength=wavelength,
        mask=mask,
        kan_model=model,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        device=device,
        p0=None,
    )
    
    plt.figure()
    plt.plot(wavelength, spec, 'k-', label='input')
    plt.plot(wavelength, model_spec, "r--", label="best-fit")
    plt.legend()
    plt.show()

    