import pandas as pd
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from astropy.io import fits
import corner
import emcee
from torch.autograd import Variable
import torch
from bayes_opt import BayesianOptimization
import nevergrad as ng

sys.path.append("/home/wangr/code/")
# from parameters.csst_parameters.csst_payne import utils, fitting
from The_Payne import utils

import conv

dtype = torch.cuda.FloatTensor
device = torch.device('cpu')

matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)

training_labels, training_spectra, validation_labels, validation_spectra = (
        utils.load_training_data()
    )

label_max = np.max(training_labels, axis=0)
label_min = np.min(training_labels, axis=0)

flux_max = np.max(training_spectra, axis=0)
flux_min = np.min(training_spectra, axis=0)

model_payne = torch.load("./model_save/Payne_KAN_model_01.kpt")
model_payne.to(device)

wave = np.load(
    "/home/wangr/code/The_Payne_KAN/The_Payne_KAN/other_data/apogee_wavelength.npz"
)
wavelength = wave["wavelength"]
spec_err = 1e-2 * np.ones(len(wavelength))
mask = np.zeros(len(wavelength), dtype=int)


def leaky_relu(z):
    return z * (z > 0) + 0.01 * z * (z < 0)


def get_spectrum_from_neural_net(scaled_labels, NN_coeffs):
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = (
        NN_coeffs
    )
    inside = np.einsum("ij,j->i", w_array_0, scaled_labels) + b_array_0
    outside = np.einsum("ij,j->i", w_array_1, leaky_relu(inside)) + b_array_1
    spectrum = np.einsum("ij,j->i", w_array_2, leaky_relu(outside)) + b_array_2
    return spectrum


def lnprob(x, obs_wave, obs_flux, obs_err, obs_mask=None):
    if obs_mask is None:
        obs_mask = np.zeros_like(obs_flux)
    idx = np.logical_and(obs_err > 0, obs_mask == 0)
    teff_range, logg_range, feh_range, afe_range = (
        [2500, 12000],
        [0, 5.5],
        [-2.5, 0.5],
        [-0.4, 0.5],
    )
    cfe_range, nfe_range, ofe_range = [-0.2, 0.5], [-0.2, 0.5], [-0.2, 0.5]
    if (
        x[0] < teff_range[0]
        or x[0] > teff_range[1]
        or x[1] < logg_range[0]
        or x[1] > logg_range[1]
        or x[2] < feh_range[0]
        or x[2] > feh_range[1]
        or x[3] < afe_range[0]
        or x[3] > afe_range[1]
        or x[4] < cfe_range[0]
        or x[4] > cfe_range[1]
        or x[5] < nfe_range[0]
        or x[5] > nfe_range[1]
        or x[6] < ofe_range[0]
        or x[6] > ofe_range[1]
    ):
        return -1e50

    scaled_label = (x - label_min) / (label_max - label_min) #- 0.5
    syn_flux = get_spectrum_from_neural_net(
        scaled_labels=scaled_label, model=model_payne
    )
    mod_flux = interp1d(wavelength, syn_flux)(obs_wave)

    # plt.figure()
    # plt.plot(obs_wave, obs_flux,'k-')
    # plt.plot(obs_wave[idx], mod_flux[idx],'r--')
    # plt.show()

    chi2 = np.mean((obs_flux[idx] - mod_flux[idx]) ** 2 / obs_err[idx] ** 2)
    print("chi: ", chi2)
    p = np.exp(-1 / 2 * chi2)
    print("prob: ", p)

    return np.log(p)


def run_mcmc(Xtest, ytest):
    # start to configure emcee
    for param, spec in zip(Xtest, ytest):
        print(param)
        try:
            wave = np.arange(2550, 10010, 10)
            flux = spec
            mask = np.zeros_like(flux)
            # p0labels = get_init_p0(spec)
            # print(p0labels)
            nwalkers = 50
            ndim = 7  # 4
            # teff0, logg0, feh0, alpha0 = [5700, 4.5, 0., 0.]
            p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
            # p0[:, 0] = p0labels[0] + 100 * p0[:, 0]
            # p0[:, 1] = p0labels[1] + 0.1 * p0[:, 1]
            # p0[:, 2] = p0labels[2] + 0.1 * p0[:, 2]
            # p0[:, 3] = p0labels[3] + 0.05 * p0[:, 3]
            # p0[:, 4] = p0labels[4] + 0.05 * p0[:, 4]
            # p0[:, 5] = p0labels[5] + 0.05 * p0[:, 5]
            # p0[:, 6] = p0labels[6] + 0.05 * p0[:, 6]
            # print(p0)
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob, args=[wave, flux, spec_err, mask]
            )
            pos, prob, state = sampler.run_mcmc(p0, 20)
            sampler.reset()
            sampler.run_mcmc(pos, 500)

            samples = sampler.chain[:, :, :].reshape((-1, ndim))
            print(samples.shape)
            fig = plt.figure(figsize=(10, 8))
            corner.corner(
                data=samples,
                labels=[
                    "$Teff$",
                    "$log g$",
                    "$[Fe/H]$",
                    r"$[\alpha/Fe]$",
                    "$[C/Fe]$",
                    "$[N/Fe]$",
                    "$[O/Fe]$",
                ],
                quantiles=[0.16, 0.5, 0.84],
                fig=fig,
                show_titles=True,
                title_kwargs={"fontsize": 14},
            )
            plt.show()

        except:  # noqa: E722
            pass
        # plt.savefig('emcee_sample.png',bbox_inches='tight')


def predict_sp_csstsls(
    wave, flux, flux_err, mask, sp_p0=None, chifit=True, mcmc=True, corner_fig_path=None
):
    flux_re = interp1d(wave[np.where(mask == 0)], flux[np.where(mask == 0)])(wavelength)
    flux_err_re = interp1d(wave[np.where(mask == 0)], flux_err[np.where(mask == 0)])(
        wavelength
    )

    scale = np.median(flux_re)
    flux = flux_re / scale
    flux_err = flux_err_re / scale
    # res_cnn = get_init_p0(flux)[:-1]


    if mcmc:
        nwalkers = 50
        ndim = 25

        if sp_p0 is None:
            teff0, logg0, feh0, alpha0 = [5777, 4.4, 0.0, 0.0]
            p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
            p0[:, 0] = teff0 + 300 * p0[:, 0]
            p0[:, 1] = logg0 + 0.3 * p0[:, 1]
            p0[:, 2] = feh0 + 0.3 * p0[:, 2]
            p0[:, 3] = alpha0 + 0.2 * p0[:, 3]
        else:
            p0 = sp_p0

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob, args=[wavelength, flux, flux_err, None]
        )
        pos, prob, state = sampler.run_mcmc(p0, 20)

        sampler.reset()
        sampler.run_mcmc(pos, 500)

        samples = sampler.chain[:, :, :].reshape((-1, ndim))

        fig = plt.figure(2, figsize=(14, 12))
        corner.corner(
            data=samples,
            labels=[
                # r"$Teff$",
                # r"logg",
                # r"[Fe/H]",
                # "$[\\alpha$/Fe]",
                # r"[C/Fe]",
                # r"[N/Fe]",
                # r"[O/Fe]",
            ],
            quantiles=[0.16, 0.5, 0.84],
            fig=fig,
            show_titles=True,
            title_kwargs={"fontsize": 14},
        )
        fig.savefig(corner_fig_path)
        # plt.show()

        sp = np.quantile(samples, 0.5, axis=0)
        err_lo = np.quantile(samples, 0.25, axis=0)
        err_up = np.quantile(samples, 0.75, axis=0)
        res_mcmc = sp
        err_lo_mcmc = err_lo
        err_up_mcmc = err_up

    return res_mcmc, err_lo_mcmc, err_up_mcmc


class obs_spec:
    def __init__(self, obs_flux, obs_wave, obs_err, obs_mask, model=None):
        self.wave = obs_wave
        self.flux = obs_flux
        self.err = obs_err
        self.mask = obs_mask
        self.model = model

    # def loss_func(self, y1, y2, y3, y4, y5, y6, y7):
    #     x = np.array([y1, y2, y3, y4, y5, y6, y7])
    #     syn_flux = get_spectrum_from_neural_net(scaled_labels=x, NN_coeffs=NN_coeffs)
    #     mod_flux = interp1d(wavelength, syn_flux)(self.wave)
    #     chi2 = np.mean(((self.flux - mod_flux) / self.err) ** 2)
    #     # print('chi: ', chi2)
    #     p = np.exp(-1 / 2 * chi2)
        # return p

    def loss_func_KANpayne(self, labels):
        # labels = np.array([y1, y2, y3])
        syn_flux = self.model(
            torch.from_numpy(labels).view(1, 25).type(dtype),
        )
        mod_flux = interp1d(wavelength, syn_flux.detached().cpu())(self.wave)
        chi2 = np.sum(((self.flux - mod_flux) / self.err) ** 2)
        return chi2
    
    def produce_spectra_by_KANpayne(self, labels):
        syn_flux = self.model(
            torch.from_numpy(labels).view(1, 25).type(dtype),
        )
        model_flux = interp1d(wavelength, syn_flux.detached().cpu())(self.wave)
        return model_flux


def pred_bayes_mcmc(Xtest, ytest):
    # Xtrain, Xtest, ytrain, ytest = read_synth_spec()
    # start to configure emcee
    bounds = np.zeros((2, 7))
    bounds[0, :] = -0.5
    bounds[1, :] = 0.5

    p0labels = [5.7, 4.5, 0.0, 0.0, -0.1, 0.2, 0]
    for param, spec in zip(Xtest, ytest):
        print("******" * 20)
        print("parameters: ", param)
        # try:
        obswave = np.arange(2550, 10010, 10)
        obsflux = spec / np.mean(spec)
        obserr = 0.02 * np.ones_like(obsflux)
        obsmask = np.zeros_like(obsflux)
        # plt.figure()
        # plt.plot(obswave, obsflux)
        # plt.show()
        obs = obs_spec(
            obs_flux=obsflux,
            obs_wave=obswave,
            obs_err=obserr,
            obs_mask=obsmask,
            model=model_payne,
        )
        # optimizer = BayesianOptimization(f=obs.loss_func,
        #  pbounds=pbounds,
        #  verbose=0)
        # optimizer.maximize(init_points=50, n_iter=505, )
        instrum = ng.p.Instrumentation(
            ng.p.Array(shape=(1, 3)).set_bounds(lower=-0.5, upper=0.5),
        )

        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100)
        # optimizer.probe(p0labels)
        optimizer.minimize(obs.loss_func_MHApayne)

        bestfit = optimizer.max["params"]
        fit_res = np.array([bestfit["y" + str(s)] for s in range(1, 8)])
        res_denorm = (fit_res + label_min) * (label_max - label_min)
        print("bayes bestfit: ", res_denorm)
        fit_res_err = np.array([list(s["params"].values()) for s in optimizer.res]).std(
            axis=0
        )
        res_err_denorm = fit_res_err * (label_max - label_min)
        print("bayes errors: ", res_err_denorm)
        # return res_denorm, res_err_denorm


if __name__ == '__main__':
    run_mcmc()
#    pred_bayes_mcmc()
