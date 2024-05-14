# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: py311
#     language: python
#     name: python3
# ---

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import sys

sys.path.append("/home/wangr/code/efficient-kan/src/")
from efficient_kan import KAN

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from The_Payne import training
from The_Payne import utils
# from kan import KAN


# %%
training_labels, training_spectra, validation_labels, validation_spectra = (
    utils.load_training_data()
)

# %%
dtype = torch.float
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cpu")

# %%
x_max = np.max(training_labels, axis=0)
x_min = np.min(training_labels, axis=0)
x = (training_labels - x_min) / (x_max - x_min)  # - 0.5
x_valid = (validation_labels - x_min) / (x_max - x_min)  # - 0.5
y_max = np.max(training_spectra, axis=0)
y_min = np.min(training_spectra, axis=0)

# %%
# dimension of the output
num_pixel = training_spectra.shape[1]
num_neurons = 3
num_features = 5
dim_in = x.shape[1]

# %%
model = torch.load("./model_save/Payne_KAN_model_01.kpt")
model.to(device)


# %%

wave = np.load(
    "/home/wangr/code/The_Payne_KAN/The_Payne_KAN/other_data/apogee_wavelength.npz"
)
wavelength = wave["wavelength"]


# %%
spec_err = 1e-2 * np.ones(len(wavelength))

# for a single-star model, the format of "labels" is [Teff, Logg, Vturb [km/s],
#              [C/H], [N/H], [O/H], [Na/H], [Mg/H],\
#              [Al/H], [Si/H], [P/H], [S/H], [K/H],\
#              [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],\
#              [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H],\
#              C12/C13, Vmacro [km/s], radial velocity (RV)
real_labels = scaled_labels = [
    5770,
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
]  # assuming RV = 3 km/s.

# scale the labels (except for RV) the same as it was done during the training of the network
scaled_labels[:-1] = (real_labels[:-1] - x_min) / (x_max - x_min) #- 0.5
print(np.array(scaled_labels).shape)

real_spec = model(
    torch.tensor(scaled_labels[:-1]).view(1, 25).type(dtype)
    ).data.numpy()[0] * (y_max - y_min) + y_min
real_spec = utils.doppler_shift(wavelength, real_spec, scaled_labels[-1])

# zoom in on a small region of the spectrum so we can see what's going on.
lambda_min, lambda_max = 16000, 16100  # for plotting
m = (wavelength < lambda_max) & (wavelength > lambda_min)

plt.figure(figsize=(14, 4))
plt.plot(wavelength[m], real_spec[m], "k", lw=0.5)
plt.xlim(lambda_min, lambda_max)
plt.ylim(0.7, 1.05)

# %%
# state = torch.load(
#     "/home/wangr/code/The_Payne_KAN/model_ckpt/Payne_KAN_model_grid100.kpt"
# )
# model.load_state_dict(state)
lambda_min, lambda_max = 15000, 17500  # for plotting
m = (wavelength < lambda_max) & (wavelength > lambda_min)

# plt.figure(figsize=(14, 4))
for i in range(10):
    flux_input = training_spectra[i]
    flux_pred = model(torch.from_numpy(x[i].reshape(1, 25)).type(dtype))
    flux_pred = flux_pred.data.numpy()[0] * (y_max - y_min) + y_min

    plt.figure(i, figsize=(14, 6))
    plt.subplot(211)
    plt.plot(wavelength[m], flux_input[m], c="k", label="raw")
    plt.plot(wavelength[m], flux_pred[m], "r--", label="pred")
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.subplot(212)
    plt.plot(
        wavelength[m],
        flux_pred[m] - flux_input[m],
        c="k",
        label="raw",
    )
    plt.xlabel("Wavelength (A)")
    plt.show()

# %%
tmp = np.load(
    "/home/wangr/code/efficient-kan/examples/training_loss.npz"
)  # the output array also stores the training and validation loss
training_loss = tmp["training_loss"]
validation_loss = tmp["validation_loss"]

plt.figure(figsize=(14, 4))
plt.plot(
    np.arange(training_loss.size),
    training_loss,
    "k",
    lw=0.5,
    label="Training set",
)
plt.plot(
    np.arange(training_loss.size),
    validation_loss,
    "r",
    lw=0.5,
    label="Validation set",
)
plt.legend(loc="best", frameon=False, fontsize=18)
plt.yscale("log")
plt.ylim([5, 10000])
# plt.xlim(0,1000)
plt.xlabel("Step", size=20)
plt.ylabel("Loss", size=20)
