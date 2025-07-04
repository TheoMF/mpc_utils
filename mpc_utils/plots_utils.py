import matplotlib.pyplot as plt
import numpy as np

from mpc_utils.plot_tails import plot_tails


def plot_values(
    title, values, time, labels=None, ylabels=None, semilogs=None, ylimits=None
):
    values = np.array(values)
    if len(values.shape) == 1:
        values = values[:, np.newaxis]
    nb_plots = values.shape[1]
    fig, ax = plt.subplots(nb_plots, 1)
    fig.canvas.manager.set_window_title(title)
    if nb_plots == 1:
        if labels is not None:
            ax.plot(time, values, label=labels[0])
        else:
            ax.plot(time, values)
        ax.legend()
        ax.set_xlabel("t (s)")
        if ylabels is not None:
            ax.set_ylabel(ylabels)
    else:
        for i in range(values.shape[1]):
            if labels is not None:
                if semilogs is not None and semilogs[i] is True:
                    ax[i].semilogy(time, values[:, i], label=labels[i])
                else:
                    ax[i].plot(time, values[:, i], label=labels[i])
            else:
                ax[i].plot(time, values[:, i])
            ax[i].legend()
            ax[i].set_xlabel("t (s)")
            if ylimits is not None:
                ax[i].set_ylim(ylimits[i][0], ylimits[i][1])
            if ylabels is not None:
                ax[i].set_ylabel(ylabels)


def per_ax_plot_values(ax, values, time, labels, ylabels, twinx_axis):
    if labels is not None:
        twinx_ax = None
        check_dimensions(labels, ylabels, values)
        if True in twinx_axis:
            twinx_ax = ax.twinx()
        for idx in range(len(labels)):
            if twinx_axis[idx]:
                twinx_ax.plot(time, values, label=labels[idx])
                if ylabels is not None:
                    twinx_ax.set_ylabel(ylabels[1])
            else:
                ax.plot(time, values, label=labels[idx])
    else:
        ax.plot(time, values)
    ax.legend()
    ax.set_xlabel("t (s)")
    if ylabels is not None:
        ax.set_ylabel(ylabels)


def check_dimensions(labels, ylabels, values):
    return values.shape[2] == len(labels) and len(labels) == len(ylabels)


def plot_values_on_same_fig(title, values, time, labels=None):
    values = np.array(values)
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(title)
    for i in range(values.shape[1]):
        if labels is not None:
            ax.plot(time, values[:, i], label=labels[i])
        else:
            ax.plot(time, values[:, i])
        ax.legend()
        ax.set_xlabel("t (s)")


def concatenate_arrays_columns(array1, array2):
    if len(array1.shape) == len(array2.shape):
        array1 = array1[:, np.newaxis]
    return np.c_[array1[: array2.shape[0]], array2[:, np.newaxis]]


def concatenate_array_with_list_of_arrays(array, list_array):
    for array_to_concatenate in list_array:
        array = np.c_[
            array[: array_to_concatenate.shape[0]], array_to_concatenate[:, np.newaxis]
        ]
    return array


def plot_x0s_diff(expected_x0s, real_x0s, time):
    x0s_diff = np.abs(real_x0s - expected_x0s)
    plot_values(
        "q0 diff",
        x0s_diff[:, :7],
        time[: x0s_diff.shape[0]],
        [f"q{i}" for i in range(1, 8)],
        semilogs=[True] * 14,
    )
    plot_values(
        "dq0 diff",
        x0s_diff[:, 7:],
        time[: x0s_diff.shape[0]],
        [f"dq{i}" for i in range(1, 8)],
        semilogs=[True] * 7,
    )


def get_time_per_iteration(nb_iters):
    time_per_iters = nb_iters.copy()
    for idx, val in enumerate(nb_iters):
        if val > 0:
            time_per_iters[idx] /= val
    return time_per_iters


def plot_mpc_data(mpc_data, mpc_config, rmodel, which_plots):
    # Computation time plots
    if "computation_time" in which_plots:
        solve_time = np.array(mpc_data["solve_time"])
        time = np.linspace(0, (solve_time.shape[0] - 1) * 0.01, solve_time.shape[0])
        # plot_mpc_iter_durations("MPC iterations duration", solve_time, time)
        plot_values("MPC iterations duration", solve_time, time)
        print("solve time mean ", np.mean(solve_time))

    # Collisions pairs distance plots
    if "collision_distance" in which_plots:
        coll_avoidance_keys = [
            val for val in list(mpc_data.keys()) if "avoid_collision" in val
        ]
        if coll_avoidance_keys == []:
            raise RuntimeError(
                f"no collision pairs distances in mpc_data dictionary, keys are {mpc_data.keys()}"
            )
        coll_distance_residuals = []
        for key in coll_avoidance_keys:

            coll_distance_residuals.append(np.array(mpc_data[key])[:, 0])
        nb_vals = len(mpc_data[coll_avoidance_keys[0]])
        coll_distance_residuals = np.array(coll_distance_residuals)
        coll_distance_residuals = coll_distance_residuals.transpose()
        time_col = np.linspace(
            0,
            (nb_vals - 1) * 0.01,
            nb_vals,
        )
        coll_labels = [f"col_term_{i}" for i in range(coll_distance_residuals.shape[0])]
        plot_values(
            "collision pairs distances", coll_distance_residuals, time_col, coll_labels
        )

    # Number of iterations and kkt norms
    if "iter" in which_plots:
        kkt_norms = np.array(mpc_data["kkt_norms"])
        nb_iters = np.array(mpc_data["nb_iters"])
        nb_qp_iters = np.array(mpc_data["nb_qp_iters"])
        per_iter_avg_times = get_time_per_iteration(nb_iters=nb_iters)
        # nb_saturated_cons = get_nb_in_saturation_constraint(col_values,safety_margin=0.02, eps=1e-3)
        concatenated_values = concatenate_array_with_list_of_arrays(
            kkt_norms, [nb_iters, nb_qp_iters, per_iter_avg_times]
        )
        time = np.linspace(0, (kkt_norms.shape[0] - 1) * 0.01, kkt_norms.shape[0])

        plot_values(
            "kkt norm and solver iterations",
            concatenated_values,
            time,
            [
                "kkt norms ",
                "nb iterations",
                "nb qp iters",
                "per_iter_avg_times",
            ],
            semilogs=[True, False, False, False],
        )

    # Visual servoing
    if "visual_servoing" in which_plots:
        w_pose = [
            next(iter(mpc_input.weights.w_end_effector_poses.values()))
            for mpc_input in mpc_data["mpc_inputs"]
        ]

        visual_servoing_state = np.zeros((len(w_pose), 1))
        for idx in range(len(w_pose)):
            if (w_pose[idx] == np.zeros(6)).all():
                visual_servoing_state[idx] = 0
            elif w_pose[idx][0] >= w_pose[idx - 1][0]:
                visual_servoing_state[idx] = 1
            else:
                visual_servoing_state[idx] = 2
        time = np.linspace(
            0,
            (visual_servoing_state.shape[0] - 1) * 0.01,
            visual_servoing_state.shape[0],
        )
        plot_values(
            "Visual servoing state",
            visual_servoing_state,
            time,
            [
                "0 : IDLE, 1: VISUAL_SERVOING_ACTIVE, 2: COMING_BACK_TO_IDLE",
            ],
        )

    # Plot predictions
    if "predictions" in which_plots:
        mpc_xs = np.array(mpc_data["states_predictions"])
        mpc_us = np.array(mpc_data["control_predictions"])

        ctrl_refs = (
            np.array(mpc_data["control_reg_references"])
            if "control_reg_references" in mpc_data.keys()
            else None
        )
        print("ctrl none ? ", ctrl_refs is None)
        state_refs = (
            np.array(mpc_data["state_reg_references"])
            if "state_reg_references" in mpc_data.keys()
            else None
        )
        print("state none ? ", state_refs is None)
        translation_refs = (
            np.array(mpc_data["goal_tracking_references"])[:, :3]
            if "goal_tracking_references" in mpc_data.keys()
            else None
        )
        plot_tails(
            mpc_xs,
            mpc_us,
            rmodel,
            mpc_config,
            ctrl_refs=ctrl_refs,
            state_refs=state_refs,
            translation_refs=translation_refs,
        )
