import torch
import numpy as np
import matplotlib.pyplot as plt

import torch


def save_model_params_to_npz(model, filepath):
    weights = []
    spline_weights = []
    spline_scalers = []
    grids = []
    spline_orders = []

    for layer in model.layers:
        weights.append(layer.base_weight.cpu().detach().numpy())
        spline_weights.append(layer.spline_weight.cpu().detach().numpy())
        if hasattr(layer, "spline_scaler"):
            spline_scalers.append(layer.spline_scaler.cpu().detach().numpy())
        grids.append(layer.grid.cpu().detach().numpy())  # 保存 grid
        spline_orders.append(layer.spline_order)

    np.savez(
        filepath,
        weights=np.array(weights, dtype=object),
        spline_weights=np.array(spline_weights, dtype=object),
        spline_scalers=np.array(spline_scalers, dtype=object),
        grids=np.array(grids, dtype=object),
        spline_orders=np.array(spline_orders, dtype=object),
    )


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def b_splines(x, grid, spline_order):
    # in_features = x.shape[1]
    grid = grid  # (in_features, grid_size + 2 * spline_order + 1)
    x = np.expand_dims(x, axis=-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(np.float32)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )
    return bases


def kan_linear(x, w, spline_weight, spline_scaler, grid, spline_order, activation=gelu):
    x = np.array(x)
    w = np.array(w)
    spline_weight = np.array(spline_weight)
    spline_scaler = np.array(spline_scaler)
    grid = np.array(grid)

    base_output = np.dot(activation(x), w.T)
    spline_bases = b_splines(x, grid, spline_order)

    scaled_spline_weight = spline_weight * spline_scaler[..., np.newaxis]
    spline_output = np.dot(
        spline_bases.reshape(x.shape[0], -1),
        scaled_spline_weight.reshape(w.shape[0], -1).T,
    )

    return base_output + spline_output


def get_spectrum_from_kan(scaled_labels, KAN_coeffs):
    (
        kan_weights,
        kan_spline_weights,
        kan_spline_scalers,
        kan_grids,
        kan_spline_orders,
    ) = KAN_coeffs
    w_array_0, w_array_1, w_array_2 = kan_weights
    spline_weight_0, spline_weight_1, spline_weight_2 = kan_spline_weights
    spline_scaler_0, spline_scaler_1, spline_scaler_2 = kan_spline_scalers
    grid_0, grid_1, grid_2 = kan_grids
    spline_order_0, spline_order_1, spline_order_2 = kan_spline_orders

    inside = kan_linear(
        scaled_labels,
        w_array_0,
        spline_weight_0,
        spline_scaler_0,
        grid_0,
        spline_order_0,
    )
    outside = kan_linear(
        inside, w_array_1, spline_weight_1, spline_scaler_1, grid_1, spline_order_1
    )
    spectrum = kan_linear(
        outside, w_array_2, spline_weight_2, spline_scaler_2, grid_2, spline_order_2
    )
    return spectrum[0]


def load_model_params_from_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    weights = data["weights"]
    spline_weights = data["spline_weights"]
    spline_scalers = data["spline_scalers"]
    grids = data["grids"]
    spline_orders = data["spline_orders"]

    return weights, spline_weights, spline_scalers, grids, spline_orders


if __name__ == "__main__":
    model = torch.load(
        "/mnt/data18/code/spec_stellar_parameter/csst_parameter/csst_kan_payne/"
        + "model_save/Payne_KAN_phoe_sc_model_02.kpt"
    )
    save_model_params_to_npz(
        model,
        "/mnt/data18/code/spec_stellar_parameter/csst_parameter/csst_kan_payne/kan_model_params.npz",
    )
    scaled_labels = np.array([[0.5, 0.5, 0.5]])  # 示例输入
    KAN_coeffs = load_model_params_from_npz(
        "/mnt/data18/code/spec_stellar_parameter/csst_parameter/csst_kan_payne/kan_model_params.npz",
    )
    spectrum1 = get_spectrum_from_kan(scaled_labels, KAN_coeffs)
    spec = (
        model(torch.from_numpy(scaled_labels).type(torch.float).cuda())
        .cpu()
        .data.numpy()[0]
    )
    plt.figure()
    # plt.plot(spec, label='model')
    plt.plot(spectrum1, label="math")
    plt.plot(spec, "r--", label="model")
    plt.legend()
    plt.show()
