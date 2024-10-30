import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import deepxde as dde

from pathlib import Path

from parameters import N, beta, sigma, S0, I0, R0

sns.set_theme(style="whitegrid")


def dinn(
    data_observed,
    parameters,
    hyperparameters,
    experiment_name
):

    t_observed = data_observed[["t"]].to_numpy()
    S_observed = data_observed[["S"]].to_numpy()
    I_observed = data_observed[["I"]].to_numpy()
    R_observed = data_observed[["R"]].to_numpy()

    # Variables
    _beta = dde.Variable(0.0)
    _sigma = dde.Variable(0.0)

    variables_list = [_beta, _sigma]

    # ODE model
    def ode(t, y):
        S = y[:, 0:1]
        I = y[:, 1:2]
        R = y[:, 2:3]

        dS_dt = dde.grad.jacobian(y, t, i=0)
        dI_dt = dde.grad.jacobian(y, t, i=1)
        dR_dt = dde.grad.jacobian(y, t, i=2)

        return [
            dS_dt - ( - _beta * S / N * I ),
            dI_dt - ( _beta * S / N * I - _sigma * I  ),
            dR_dt - ( _sigma * I ),
            N - ( S + I + R)
        ]

    # Geometry
    geom = dde.geometry.TimeDomain(t_observed[0, 0], t_observed[-1, 0])

    # Boundaries
    def boundary(_, on_initial):
        return on_initial

    # Observed data
    observed_S = dde.icbc.PointSetBC(t_observed, S_observed, component=0)
    observed_I = dde.icbc.PointSetBC(t_observed, I_observed, component=1)
    observed_R = dde.icbc.PointSetBC(t_observed, R_observed, component=2)

    # Model
    data = dde.data.PDE(
        geom,
        ode,
        [
            observed_S,
            observed_I,
            observed_R
        ],
        num_domain=0,
        num_boundary=2,
        anchors=t_observed,
    )

    neurons = hyperparameters["neurons"]
    layers = hyperparameters["layers"]
    activation = hyperparameters["activation"]
    net = dde.nn.FNN([1] + [neurons] * layers + [3], activation, "Glorot uniform")

    def feature_transform(t):
        t = t / t_observed[-1, 0]
        return t

    # net.apply_feature_transform(feature_transform)


    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        external_trainable_variables=variables_list
    )

    variable_filename = output_path / f"{experiment_name}.dat"
    variable = dde.callbacks.VariableValue(
        [_beta, _sigma],
        period=100,
        filename=variable_filename
    )

    losshistory, train_state = model.train(
        iterations=hyperparameters["iterations"],
        display_every=10000,
        callbacks=[variable]
    )


    parameters_pred = {
        name: var
        for name, var in zip(parameters.keys(), variable.value)
    }
    return model, losshistory, train_state, parameters_pred, variable


if __name__ == "__main__":

    if len(sys.argv) > 1:
        experiment_id = str(sys.argv[1])
        experiment_name = f"sir_{experiment_id}"
    else:
        experiment_name = "sir"

    parameters_real = {
        "beta": beta,
        "sigma": sigma
    }

    hyperparameters = {
        "iterations": 30000,
        "layers": 3,
        "neurons": 32,
        "activation": "relu"
    }

    data_path = Path().resolve().parent / "data"
    output_path = Path().resolve().parent / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    data_observed = pd.read_csv(data_path / "sir_noise1.csv")
    model, losshistory, train_state, parameters_pred, variable = dinn(
        data_observed=data_observed,
        parameters=parameters_real,
        hyperparameters=hyperparameters,
        experiment_name=experiment_name
    )

    lines = open(output_path / f"{experiment_name}.dat", "r").readlines()
    raw_parameters_pred_history = np.array(
        [
            np.fromstring(
                min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                sep=",",
            )
            for line in lines
        ]
    )

    iterations = [int(re.findall("^[0-9]+", line)[0]) for line in lines]

    parameters_pred_history = {
        name: raw_parameters_pred_history[:, i]
        for i, (name, _) in enumerate(parameters_real.items())
    }

    n_callbacks, n_variables = raw_parameters_pred_history.shape
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 5))

    ax1.plot(iterations, parameters_pred_history["beta"] , "-")
    ax1.plot(iterations, np.ones_like(iterations) * parameters_real["beta"], "--")
    ax1.set_ylabel(r"$\beta$")

    ax2.plot(iterations, parameters_pred_history["sigma"] , "-")
    ax2.plot(iterations, np.ones_like(iterations) * parameters_real["sigma"], "--")
    ax2.set_ylabel(r"$\sigma$")
    ax2.set_xlabel("Iterations")

    fig.tight_layout()
    fig.savefig(output_path / f"{experiment_name}_parameter_estimation.png", dpi=300)
    plt.close()

    error_df = (
        pd.DataFrame(
            {
                "Real": parameters_real,
                "Predicted": parameters_pred
            }
        )
        .assign(
            **{"Relative Error": lambda x: (x["Real"] - x["Predicted"]).abs() / x["Real"]}
        )
    )
    error_df.to_csv(output_path / f"{experiment_name}_relative_error.csv")