import numpy as np
import scipy
from scipy import interpolate, signal
from scipy.optimize import least_squares
from scipy.stats import norm
import nevergrad as ng
import torch
import sys

import util, spectral_model, least_squares_fit, fitting
import os


_PATH = os.path.dirname(__file__)
# from parameters.medres_parameter.normalization.normalize_cannon import pseudo_continuum
dtype = torch.cuda.FloatTensor
bias_spec, err_spec = np.load(
    os.path.jpin(_PATH, "..", "data", "bias_std_res_spec_transformerPayne.npy")
)


class obs_spec:
    def __init__(
        self,
        obs_flux: np.ndarray,
        obs_wave: np.ndarray,
        obs_err: np.ndarray,
        obs_mask: np.ndarray = None,
        model=None,
        model_err: np.ndarray = None,
    ):
        """_summary_

        Parameters
        ----------
        obs_flux : np.ndarray
            _description_
        obs_wave : np.ndarray
            _description_
        obs_err : np.ndarray
            _description_
        obs_mask : np.ndarray, optional
            _description_, by default None
        model : _type_, optional
            _description_, by default None
        model_err : np.ndarray, optional
            _description_, by default None
        """
        self.wavelength_model = np.arange(2500, 10000, 2)
        self.wave = obs_wave
        self.flux = obs_flux
        self.err = obs_err
        self.mask = obs_mask
        self.model = model
        if model_err is None:
            model_err = err_spec  # np.zeros_like(self.wavelength_model)
        self.model_err = model_err

    def loss_func_KANpayne(self, labels: np.array):
        """_summary_

        Parameters
        ----------
        labels : np.array
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # labels = np.array([teff, logg, feh])
        syn_flux = (
            self.model(
                torch.arange(0, 3750).view(1, 3750).type(dtype),
                torch.from_numpy(labels[:3]).view(1, 3).type(dtype),
            )
            .cpu()
            .detach()
            .numpy()
        )

        mod_flux = interpolate.interp1d(self.wavelength_model, syn_flux)(self.wave)
        mod_flux_err2 = interpolate.interp1d(self.wavelength_model, self.model_err**2)(
            self.wave
        )

        # win_width =  int((labels[-1] + 0.5) * 20) + 1
        # win = signal.windows.hann(win_width)

        # mod_flux = signal.convolve(mod_flux.squeeze(), win, mode='same')/sum(win)
        # mod_flux[:int(win_width/2)]=1
        # mod_flux[-int(win_width/2):]=1
        # mod_flux_err2 = signal.convolve(mod_flux_err2.squeeze(), win, mode='same')/sum(win)
        mod_flux = scipy.ndimage.gaussian_filter1d(mod_flux.squeeze(), sigma=1.97)
        mod_flux_err2 = scipy.ndimage.gaussian_filter1d(
            mod_flux_err2.squeeze(), sigma=1.97
        )
        chi2 = np.mean(
            (self.flux.squeeze() - mod_flux) ** 2
            / (self.err.squeeze() ** 2 + mod_flux_err2)
        )

        return chi2

    def produce(self, labels: np.array):
        """_summary_

        Parameters
        ----------
        labels : np.array
            _description_

        Returns
        -------
        _type_
            _description_
        """
        syn_flux = (
            self.model(
                torch.arange(0, 3750).view(1, 3750).type(dtype),
                torch.from_numpy(labels[:3]).view(1, 3).type(dtype),
            )
            .cpu()
            .detach()
            .numpy()
        )

        mod_flux = interpolate.interp1d(self.wavelength_model, syn_flux)(self.wave)
        mod_flux_err2 = interpolate.interp1d(self.wavelength_model, self.model_err**2)(
            self.wave
        )
        # win_width = int((labels[-1] + 0.5) * 20) + 1
        # win = signal.windows.hann(win_width)
        # mod_flux = signal.convolve(mod_flux.squeeze(), win, mode='same')/sum(win)
        # mod_flux[:int(win_width/2)]=1
        # mod_flux[-int(win_width/2):]=1
        # mod_flux_err2 = signal.convolve(mod_flux_err2.squeeze(), win, mode='same')/sum(win)
        mod_flux = scipy.ndimage.gaussian_filter1d(mod_flux.squeeze(), sigma=1.97)
        mod_flux_err2 = scipy.ndimage.gaussian_filter1d(
            mod_flux_err2.squeeze(), sigma=1.97
        )
        return mod_flux, mod_flux_err2


def fit_global(
    spectrum_obs,
    spectrum_err,
    wavelength_obs,
    NN_coeffs,
    NN_coeffs_sc,
    model_payne,
    wavelength_payne,
    errors_payne=None,
    RV_array=np.linspace(-1, 1.0, 6),
    order_choice=[1],
    polynomial_order=6,
    bounds_set=None,
    initial_stellar_parameters=None,
):
    # ---------- NN-based payne with poly model fit---------------------------
    (
        popt_best,
        pcov,
        model_spec_best,
    ) = fitting.fit_normalized_spectrum_single_star_model(
        wave=wavelength_obs.squeeze(),
        norm_spec=spectrum_obs.squeeze(),
        spec_err=spectrum_err.squeeze(),
        NN_coeffs=NN_coeffs_sc,
        wavelength_payne=wavelength_payne,
        p0=None,
    )

    chi_square = cal_ksq(model_spec_best, spectrum_obs, spectrum_err)
    # popt_best, model_spec_best, chi_square = fitting_csst_byNNcoefs(
    #                                               spectrum_obs,
    #                                               spectrum_err,
    #                                               spectrum_blaze,
    #                                               wavelength_obs,
    #                                               NN_coeffs,
    #                                             #   model_payne,
    #                                               wavelength_payne,
    #                                               errors_payne,
    #                                               p0_initial=None,
    #                                               RV_prefit=False,
    #                                               blaze_normalized=False,
    #                                               RV_array=RV_array,
    #                                               polynomial_order=polynomial_order,
    #                                               bounds_set=bounds_set,
    #                                               order_choice=order_choice)
    print("=== First NN fit res: ", popt_best, "ksq =", chi_square)
    # print('=== first NN fit res: ', utils.denormalize_stellar_parameter_labels(
    # popt_best[:3], x_min=NN_coeffs[-2], x_max=NN_coeffs[-1]), "ksq =",
    # chi_square)
    # num_labels = 3
    # RV_array = np.array([popt_best[-1]])
    # if initial_stellar_parameters is not None:
    #     normalized_stellar_parameters = utils.normalize_stellar_parameter_labels(initial_stellar_parameters, NN_coeffs=None)
    #     num_order = spectrum_obs.shape[0]
    #     coeff_poly = polynomial_order + 1
    #     p0_initial = np.zeros(num_labels + coeff_poly * num_order + 1)
    #     p0_initial[:num_labels] = normalized_stellar_parameters
    #     p0_initial[num_labels::coeff_poly] = 1
    #     p0_initial[num_labels+1::coeff_poly] = 0
    #     p0_initial[num_labels+2::coeff_poly] = 0
    #     # p0_initial[-2] = 0.5
    #     p0_initial[-1] = np.array([popt_best[-1]])
    # else:
    # p0_initial = None

    # ------------ transform-based payne never grad opt-----------------------
    c = 2.99792458e5  # km/s
    # rv = popt_best[-1] * 100
    rv = popt_best[-1]
    rv_err = np.sqrt(np.diagonal(pcov)[-1])
    doppler_factor = np.sqrt((1 - rv / c) / (1 + rv / c))
    # conti, spectrum_sc, norm_ivar = pseudo_continuum(
    #                                     wavelength=wavelength_obs.squeeze(),
    #                                     flux=spectrum_obs.squeeze(),
    #                                     ivar=1/spectrum_err.squeeze()**2,
    #                                     L=125)
    # spec_err_sc = spectrum_err/conti

    res_ng_denorm = fit_nevergrad(
        wave=wavelength_obs * doppler_factor,
        spec=spectrum_obs,
        spec_err=spectrum_err,
        model_payne=model_payne,
        x_min=NN_coeffs[-2],
        x_max=NN_coeffs[-1],
    )
    # print("res_ng_denorm: ",res_ng_denorm)
    modelspec_bestfit_ng = (
        model_payne(
            torch.arange(0, 3750).view(1, 3750).type(dtype),
            torch.from_numpy(util.normalize_stellar_parameter_labels(res_ng_denorm[:3]))
            .view(1, 3)
            .type(dtype),
        )
        .detach()
        .cpu()
        .numpy()
        .reshape(
            3750,
        )
    )

    mod_flux = interpolate.interp1d(wavelength_payne, modelspec_bestfit_ng)(
        wavelength_obs * doppler_factor
    )
    mod_flux_err2 = interpolate.interp1d(wavelength_payne, err_spec**2)(
        wavelength_obs * doppler_factor
    )

    # win = signal.windows.hann(int(res_ng_denorm[-1]))
    # bestfit_flux_ng = signal.convolve(mod_flux.squeeze(), win, mode='same')/sum(win)
    # bestfit_flux_ng[:int(res_ng_denorm[-1]/2)] = 1
    # bestfit_flux_ng[-int(res_ng_denorm[-1]/2):] = 1
    # bestfit_fluxerr2_ng = signal.convolve(mod_flux_err2.squeeze(), win, mode='same')/sum(win)
    bestfit_flux_ng = scipy.ndimage.gaussian_filter1d(mod_flux.squeeze(), sigma=1.97)
    bestfit_fluxerr2_ng = scipy.ndimage.gaussian_filter1d(
        mod_flux_err2.squeeze(), sigma=1.97
    )
    chi_value1 = np.mean(
        (spectrum_obs - bestfit_flux_ng) ** 2
        / (spectrum_err.squeeze() ** 2 + bestfit_fluxerr2_ng)
    )
    print("=== second fit res: ", res_ng_denorm, "ksq =", chi_value1)

    # ----------- continuum normalized single star NNcoeffs model-fit---------
    if 0:
        (
            popt,
            pcov,
            model_spec_bestfit,
        ) = fitting.fit_normalized_spectrum_single_star_model(
            wave=wavelength_obs.squeeze(),
            norm_spec=spectrum_obs.squeeze(),
            spec_err=spectrum_err.squeeze(),
            NN_coeffs=NN_coeffs_sc,
            wavelength_payne=wavelength_payne,
            p0=None,
        )

        ksq_value2 = cal_ksq(model_spec_bestfit, spectrum_obs, spectrum_err)
        print("=== third fit res: ", popt, "ksq =", ksq_value2)

        return (
            res_ng_denorm,
            bestfit_flux_ng,
            rv,
            chi_value1,
            popt,
            pcov,
            model_spec_bestfit,
            ksq_value2,
        )
    else:
        return res_ng_denorm, bestfit_flux_ng, rv, rv_err, chi_value1


def contimuun_normalize_syn(wave, spec, L=50):
    wave_model = np.arange(wave.min(), wave.max(), 2)

    if wave_model.max() < wave.max():
        wave_model = np.append(wave_model, wave.max())

    # print(wave.shape, spec.shape, wave_model.shape)
    flux_resample = interpolate.interp1d(wave.squeeze(), spec.squeeze())(wave_model)
    conti = scipy.ndimage.gaussian_filter1d(flux_resample, L)
    conti_resample = interpolate.interp1d(wave_model, conti)(wave.squeeze()).reshape(
        1, -1
    )
    spec_sc = spec / conti_resample
    return spec_sc, conti_resample


def fit_least_squares(
    wave, spec, spec_err, p0_initial, model_payne, wave_model, x_min, x_max
):
    obsflux = torch.from_numpy(spec).type(dtype)
    obserr = torch.from_numpy(spec_err).type(dtype)

    if spec_err is None:
        obserr = torch.ones_like(obsflux).type(dtype)

    x_weight = 1.0 / obserr

    bounds = (
        torch.from_numpy(np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]))
        .type(dtype)
        .cuda()
    )
    params = torch.nn.Parameter(torch.from_numpy(p0_initial), requires_grad=True)

    loss_full = least_squares_fit.least_squares_fit(
        model=model_payne,
        wave_model=torch.from_numpy(wave_model).type(dtype),
        spec_obs=obsflux,
        wave_obs=torch.from_numpy(wave).squeeze().type(dtype),
        weight_obs=x_weight,
        params=params,
        bounds=bounds,
    )

    print("transPayne final loss : ", loss_full)
    y_pred = util.denormalize_stellar_parameter_labels(
        params.cpu().detach().numpy(), x_min=x_min, x_max=x_max
    )

    return y_pred


def fit_nevergrad(wave, spec, spec_err, model_payne, x_min, x_max, ):
    obs = obs_spec(
        obs_flux=spec, obs_wave=wave, obs_err=spec_err, obs_mask=None, model=model_payne
    )
    instrum = ng.p.Instrumentation(
        ng.p.Array(shape=(4,)).set_bounds(lower=-0.5, upper=0.5),
    )
    optimizer_ng = ng.optimizers.NGOpt(
        parametrization=instrum, budget=1000, num_workers=8
    )
    # optimizer_ng = ng.optimizers.TwoPointsDE(
    #     parametrization=instrum,
    #     budget=1000,
    #     num_workers=8)
    recommond = optimizer_ng.minimize(obs.loss_func_KANpayne)
    # print(recommond)
    res_ng_denorm = np.zeros(shape=(4,))
    res_ng_denorm[:3] = util.denormalize_stellar_parameter_labels(
        recommond.value[0][0][:3], x_min=x_min, x_max=x_max
    )
    # win_width = int((res_ng_denorm[-1] + 0.5) * (11 - 1)) + 1
    # The code is attempting to assign the value of `win_width` to the last element of the list
    # `res_ng_denorm`. However, the code seems to be commented out with `#`, so it will not be
    # executed.
    # res_ng_denorm[-1] = win_width
    # print("=== nevergrad fit res: ", res_ng_denorm)
    return res_ng_denorm


def cal_ksq(a, b, err):
    return np.mean((a - b) ** 2 / err**2)


def fit_continuum(
    spectrum,
    spectrum_err,
    wavelength,
    previous_poly_fit,
    previous_model_spec,
    polynomial_order=1,
    previous_polynomial_order=2,
):
    """
    Fit the continuum while fixing other stellar labels
    The end results will be used as initial condition in the global fit (continuum + stellar labels)
    """
    # print('Pre Fit: Finding the best continuum initialization')
    num_labels = 3
    # normalize wavelength grid
    wavelength_normalized = util.whitten_wavelength(wavelength) * 100.0

    # number of polynomial coefficients
    coeff_poly = polynomial_order + 1
    pre_coeff_poly = previous_polynomial_order + 1

    # initiate results array for the polynomial coefficients
    fit_poly = np.zeros((wavelength_normalized.shape[0], coeff_poly))

    # loop over all order and fit for the polynomial (weighted by the error)
    for k in range(wavelength_normalized.shape[0]):
        pre_poly = 0
        for m in range(pre_coeff_poly):
            pre_poly += (wavelength_normalized[k, :] ** m) * previous_poly_fit[
                num_labels + m + pre_coeff_poly * k
            ]
        substract_factor = (
            previous_model_spec[k, :] / pre_poly
        )  # subtract away the previous fit
        fit_poly[k, :] = np.polyfit(
            wavelength_normalized[k, :],
            spectrum[k, :] / substract_factor,
            polynomial_order,
            w=1.0 / (spectrum_err[k, :] / substract_factor),
        )[::-1]

    return fit_poly


def evaluate_model(
    labels,
    model_payne,
    wavelength_payne,
    errors_payne,
    coeff_poly,
    wavelength_obs,
    num_order,
    num_pixel,
    wavelength_normalized=None,
):
    # get wavelength_normalized
    if wavelength_normalized is None:
        wavelength_normalized = util.whitten_wavelength(wavelength_obs) * 100.0
    # num_label=NN_coeffs[0].shape[1]
    num_labels = 3
    # make payne models
    full_spec = (
        model_payne(
            torch.arange(0, 3750).view(1, 3750).type(dtype),
            torch.from_numpy(labels[:num_labels]).view(1, 3).type(dtype),
        )
        .detach()
        .cpu()
        .numpy()
    )
    # broadening kernel
    # win = norm.pdf((np.arange(21) - 10.) * (wavelength_payne[1] - wavelength_payne[0]), scale=labels[-2] / 3e5 * 5000)
    # win = win / np.sum(win)

    # vbroad -> RV
    # full_spec = signal.convolve(full_spec, win, mode='same')
    full_spec = util.doppler_shift(
        wavelength_payne, full_spec, labels[-1] * 100.0, wavelength_payne
    )
    errors_spec = util.doppler_shift(
        wavelength_payne, errors_payne, labels[-1] * 100.0, wavelength_payne
    )

    # interpolate into the observed wavelength
    f_flux_spec = interpolate.interp1d(wavelength_payne, full_spec)
    f_errs_spec = interpolate.interp1d(wavelength_payne, errors_spec)

    # loop over all orders
    spec_predict = np.zeros(num_order * num_pixel)
    errs_predict = np.zeros(num_order * num_pixel)

    for k in range(num_order):
        scale_poly = 0
        for m in range(coeff_poly):
            scale_poly += (wavelength_normalized[k, :] ** m) * labels[
                num_labels + coeff_poly * k + m
            ]
        spec_predict[k * num_pixel : (k + 1) * num_pixel] = scale_poly * f_flux_spec(
            wavelength_obs[k, :]
        )
        errs_predict[k * num_pixel : (k + 1) * num_pixel] = scale_poly * f_errs_spec(
            wavelength_obs[k, :]
        )

    return spec_predict, errs_predict


def fitting_csst(
    spectrum_obs,
    spectrum_err,
    # spectrum_blaze,
    wavelength_obs,
    model_payne,
    wavelength_payne,
    errors_payne=None,
    p0_initial=None,
    bounds_set=None,
    RV_prefit=False,
    # blaze_normalized=False,
    RV_array=np.linspace(-1, 1.0, 6),
    polynomial_order=2,
    order_choice=[20],
):
    """
    Fitting spectrum
    Fitting radial velocity can be very multimodal. The best strategy is to initalize over
    different RVs. When RV_prefit is true, we first fit a single order to estimate
    radial velocity that we will adopt as the initial guess for the global fit.

    RV_array is the range of RV that we will consider
    RV array is in the unit of 100 km/s
    order_choice is the order that we choose to fit when RV_prefit is TRUE
    When blaze_normalized is True, we first normalize spectrum with the blaze
    Returns:
        Best fitted parameter (Teff, logg, Fe/H, polynomial coefficients, vmacro, RV)
    """

    # assume no model error if not specified
    if errors_payne is None:
        errors_payne = np.zeros_like(wavelength_payne)

    # normalize wavelength grid
    wavelength_normalized = util.whitten_wavelength(wavelength_obs) * 100.0
    # number of polynomial coefficients
    coeff_poly = polynomial_order + 1

    # specify a order for the (pre-) RV fit
    if RV_prefit:
        spectrum_obs = spectrum_obs[order_choice, :]
        spectrum_err = spectrum_err[order_choice, :]
        # spectrum_blaze = spectrum_blaze[order_choice, :]
        wavelength_normalized = wavelength_normalized[order_choice, :]
        wavelength_obs = wavelength_obs[order_choice, :]

    # normalize spectra with the blaze function
    # if blaze_normalized:
    #     spectrum_obs = spectrum_obs / spectrum_blaze
    #     spectrum_err = spectrum_err / spectrum_blaze

    # number of pixel per order, number of order
    num_pixel = spectrum_obs.shape[1]
    num_order = spectrum_obs.shape[0]

    # the objective function
    def fit_func(labels):
        spec_predict, errs_predict = evaluate_model(
            labels,
            model_payne,
            wavelength_payne,
            errors_payne,
            coeff_poly,
            wavelength_obs,
            num_order,
            num_pixel,
            wavelength_normalized,
        )

        # Calculate resids: set all potentially bad errors to 999.
        # We take errs > 300 as bad to account for interpolation issues on the
        # mask
        errs = np.sqrt(errs_predict**2 + spectrum_err.ravel() ** 2)
        errs[(~np.isfinite(errs)) | (errs > 300) | (errs < 0)] = 999.0
        resids = (spectrum_obs.ravel() - spec_predict) / errs
        return resids

    # ------------------------------------------------------------------------------------------
    # loop over all possible
    chi_2 = np.inf

    # if RV_prefit:
    #     print('Pre Fit: Finding the best radial velocity initialization')

    # if not RV_prefit and blaze_normalized:
    #     print('Pre Fit: Fitting the blaze-normalized spectrum')

    # if not RV_prefit and not blaze_normalized:
    #     # print('Final Fit: Fitting the whole spectrum with all parameters simultaneously')
    #     popt_for_printing = utils.transform_coefficients(
    #                                     p0_initial,
    #                                     x_max=np.array([9.8e+03, 6.0e+00, 5.0e-01]),
    #                                     x_min=np.array([3.1e+03, -5.0e-01, -4.0e+00]),
    #                                     order=3)
    # print('p0: Teff={:.0f} logg={:.2f} FeH={:.2f} vbroad={:.2f} rv={:.1f}'.format(
    #     *[popt_for_printing[i] for i in [0, 1, 2, -2, -1]]))

    num_labels = 3
    for i in range(RV_array.size):
        # print(i + 1, "/", RV_array.size)
        # initialize the parameters (Teff, logg, Fe/H, alpha/Fe, polynomial
        # continuum, vbroad, RV)
        if p0_initial is None:
            p0 = np.zeros(num_labels + coeff_poly * num_order + 1 + 1)
            p0[num_labels::coeff_poly] = 1
            p0[num_labels + 1 :: coeff_poly] = 0
            p0[num_labels + 2 :: coeff_poly] = 0
            # p0[-2] = 0.5
            p0[-1] = RV_array[i]
        else:
            p0 = p0_initial

        # set fitting bound
        bounds = np.zeros((2, p0.size))
        bounds[0, num_labels:] = -1000  # polynomial coefficients
        bounds[1, num_labels:] = 1000
        if bounds_set is None:
            bounds[0, :num_labels] = -0.5  # teff, logg, feh, alphafe
            bounds[1, :num_labels] = 0.5
            # bounds[0, -2] = 0.1  # vbroad
            # bounds[1, -2] = 10.
            bounds[0, -1] = -5.0  # RV [100 km/s]
            bounds[1, -1] = 5.0
        else:
            bounds[:, :num_labels] = bounds_set[:, :num_labels]
            # bounds[:, -2:] = bounds_set[:, -2:]

        if (not (bounds_set is None)) and (p0_initial is None):
            p0[:num_labels] = np.mean(bounds_set[:, :num_labels], axis=0)

        # run the optimizer
        tol = 5e-4
        # popt, pcov = curve_fit(fit_func, xdata=[],\
        #                       ydata = spectrum.ravel(), sigma = spectrum_err.ravel(),\
        #                       p0 = p0, bounds=bounds, ftol = tol, xtol = tol, absolute_sigma = True,\
        #                       method = 'trf')
        res = least_squares(
            fit_func, p0, bounds=bounds, ftol=tol, xtol=tol, method="trf"
        )
        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)
        popt = res.x

        # calculate chi^2
        model_spec, model_errs = evaluate_model(
            popt,
            model_payne,
            wavelength_payne,
            errors_payne,
            coeff_poly,
            wavelength_obs,
            num_order,
            num_pixel,
            wavelength_normalized,
        )

        chi_2_temp = np.mean(
            (spectrum_obs.ravel() - model_spec) ** 2
            / (model_errs**2 + spectrum_err.ravel() ** 2)
        )

        # check if this gives a better fit
        if chi_2_temp < chi_2:
            chi_2 = chi_2_temp
            model_spec_best = model_spec
            popt_best = popt

    return popt_best, model_spec_best.reshape(num_order, num_pixel), chi_2


def evaluate_NNmodel(
    labels,
    NN_coeffs,
    wavelength_payne,
    errors_payne,
    coeff_poly,
    wavelength,
    num_order,
    num_pixel,
    wavelength_normalized=None,
):
    # get wavelength_normalized
    if wavelength_normalized is None:
        wavelength_normalized = util.whitten_wavelength(wavelength) * 100.0
    num_label = NN_coeffs[0].shape[1]
    # make payne models
    full_spec = spectral_model.get_spectrum_from_neural_net(
        scaled_labels=labels[:num_label], NN_coeffs=NN_coeffs
    )
    # broadening kernel
    # win = norm.pdf((np.arange(21) - 10.) * (wavelength_payne[1] - wavelength_payne[0]), scale=labels[-2] / 3e5 * 5000)
    # win = win / np.sum(win)

    # vbroad -> RV
    # full_spec = signal.convolve(full_spec, win, mode='same')
    full_spec = util.doppler_shift(
        wavelength_payne, full_spec, labels[-1] * 100.0, wavelength_payne
    )
    errors_spec = util.doppler_shift(
        wavelength_payne, errors_payne, labels[-1] * 100.0, wavelength_payne
    )

    # interpolate into the observed wavelength
    f_flux_spec = interpolate.interp1d(wavelength_payne, full_spec)
    f_errs_spec = interpolate.interp1d(wavelength_payne, errors_spec)

    # loop over all orders
    spec_predict = np.zeros(num_order * num_pixel)
    errs_predict = np.zeros(num_order * num_pixel)

    for k in range(num_order):
        scale_poly = 0
        for m in range(coeff_poly):
            scale_poly += (wavelength_normalized[k, :] ** m) * labels[
                3 + coeff_poly * k + m
            ]
        spec_predict[k * num_pixel : (k + 1) * num_pixel] = scale_poly * f_flux_spec(
            wavelength[k, :]
        )
        errs_predict[k * num_pixel : (k + 1) * num_pixel] = scale_poly * f_errs_spec(
            wavelength[k, :]
        )

    return spec_predict, errs_predict


def fitting_csst_byNNcoefs(
    spectrum_obs,
    spectrum_err,
    # spectrum_blaze,
    wavelength_obs,
    NN_coeffs,
    wavelength_payne,
    errors_payne=None,
    p0_initial=None,
    bounds_set=None,
    RV_prefit=False,
    # blaze_normalized=False,
    RV_array=np.linspace(-1, 1.0, 6),
    polynomial_order=2,
    order_choice=[20],
):
    """
    Fitting spectrum
    Fitting radial velocity can be very multimodal. The best strategy is to initalize over
    different RVs. When RV_prefit is true, we first fit a single order to estimate
    radial velocity that we will adopt as the initial guess for the global fit.

    RV_array is the range of RV that we will consider
    RV array is in the unit of 100 km/s
    order_choice is the order that we choose to fit when RV_prefit is TRUE
    When blaze_normalized is True, we first normalize spectrum with the blaze
    Returns:
        Best fitted parameter (Teff, logg, Fe/H, polynomial coefficients, RV)
    """

    # assume no model error if not specified
    if errors_payne is None:
        errors_payne = np.zeros_like(wavelength_payne)

    # normalize wavelength grid
    wavelength_normalized = util.whitten_wavelength(wavelength_obs) * 100.0
    # number of polynomial coefficients
    coeff_poly = polynomial_order + 1

    # specify a order for the (pre-) RV fit
    if RV_prefit:
        spectrum_obs = spectrum_obs[order_choice, :]
        spectrum_err = spectrum_err[order_choice, :]
        # spectrum_blaze = spectrum_blaze[order_choice, :]
        wavelength_normalized = wavelength_normalized[order_choice, :]
        wavelength_obs = wavelength_obs[order_choice, :]

    # normalize spectra with the blaze function
    # if blaze_normalized:
    #     spectrum_obs = spectrum_obs / spectrum_blaze
    #     spectrum_err = spectrum_err / spectrum_blaze

    # number of pixel per order, number of order
    num_pixel = spectrum_obs.shape[1]
    num_order = spectrum_obs.shape[0]

    # the objective function
    def fit_func(labels):
        spec_predict, errs_predict = evaluate_NNmodel(
            labels,
            NN_coeffs,
            wavelength_payne,
            errors_payne,
            coeff_poly,
            wavelength_obs,
            num_order,
            num_pixel,
            wavelength_normalized,
        )

        # Calculate resids: set all potentially bad errors to 999.
        # We take errs > 300 as bad to account for interpolation issues on the
        # mask
        errs = np.sqrt(errs_predict**2 + spectrum_err.ravel() ** 2)
        errs[(~np.isfinite(errs)) | (errs > 300) | (errs < 0)] = 999.0
        resids = (spectrum_obs.ravel() - spec_predict) / errs
        return resids

    # ------------------------------------------------------------------------------------------
    # loop over all possible
    chi_2 = np.inf

    # if RV_prefit:
    #     print('Pre Fit: Finding the best radial velocity initialization')

    # if not RV_prefit and blaze_normalized:
    #     print('Pre Fit: Fitting the blaze-normalized spectrum')
    # print(p0_initial)
    # if not RV_prefit and not blaze_normalized:
    #     # print('Final Fit: Fitting the whole spectrum with all parameters simultaneously')
    #     popt_for_printing = utils.transform_coefficients(
    #                                     p0_initial,
    #                                     x_max=np.array([9.8e+03, 6.0e+00, 5.0e-01]),
    #                                     x_min=np.array([3.1e+03, -5.0e-01, -4.0e+00]),
    #                                     order=3)
    #     print('p0 = Teff={:.0f} logg={:.2f} FeH={:.2f} rv={:.1f}'.format(
    #         *[popt_for_printing[i] for i in [0, 1, 2, -1]]))

    num_labels = 3
    for i in range(RV_array.size):
        # print(i + 1, "/", RV_array.size)
        # initialize the parameters (Teff, logg, Fe/H, alpha/Fe, polynomial
        # continuum, vbroad, RV)
        if p0_initial is None:
            p0 = np.zeros(num_labels + coeff_poly * num_order + 1 + 1)
            p0[num_labels::coeff_poly] = 1
            p0[num_labels + 1 :: coeff_poly] = 0
            p0[num_labels + 2 :: coeff_poly] = 0
            p0[-1] = RV_array[i]
        else:
            p0 = p0_initial

        # set fitting bound
        bounds = np.zeros((2, p0.size))
        bounds[0, num_labels:] = -1000  # polynomial coefficients
        bounds[1, num_labels:] = 1000
        if bounds_set is None:
            bounds[0, :num_labels] = -0.5  # teff, logg, feh, alphafe
            bounds[1, :num_labels] = 0.5
            bounds[0, -1] = -5.0  # RV [100 km/s]
            bounds[1, -1] = 5.0
        else:
            bounds[:, :num_labels] = bounds_set[:, :num_labels]
            # bounds[:, -2:] = bounds_set[:, -2:]

        if (not (bounds_set is None)) and (p0_initial is None):
            p0[:num_labels] = np.mean(bounds_set[:, :num_labels], axis=0)

        # run the optimizer
        tol = 5e-4
        # popt, pcov = curve_fit(fit_func, xdata=[],\
        #                       ydata = spectrum.ravel(), sigma = spectrum_err.ravel(),\
        #                       p0 = p0, bounds=bounds, ftol = tol, xtol = tol, absolute_sigma = True,\
        #                       method = 'trf')
        res = least_squares(
            fit_func, p0, bounds=bounds, ftol=tol, xtol=tol, method="trf"
        )
        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)
        popt = res.x

        # calculate chi^2
        model_spec, model_errs = evaluate_NNmodel(
            popt,
            NN_coeffs,
            wavelength_payne,
            errors_payne,
            coeff_poly,
            wavelength_obs,
            num_order,
            num_pixel,
            wavelength_normalized,
        )

        chi_2_temp = np.mean(
            (spectrum_obs.ravel() - model_spec) ** 2
            / (model_errs**2 + spectrum_err.ravel() ** 2)
        )

        # check if this gives a better fit
        if chi_2_temp < chi_2:
            chi_2 = chi_2_temp
            model_spec_best = model_spec
            popt_best = popt

    return popt_best, model_spec_best.reshape(num_order, num_pixel), chi_2
