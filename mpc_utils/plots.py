import matplotlib.pyplot as plt
import numpy as np


def plot_mpc_iter_durations(title, mpc_iter_durations, time):
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(title)
    ax.plot(time, mpc_iter_durations)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("MPC iteration duration (s)")


def plot_2_mpc_iter_durations(
    title, mpc_iter_durations_1, label1, mpc_iter_durations_2, label2, time
):
    length = min(
        np.array(mpc_iter_durations_1).shape[0],
        np.array(mpc_iter_durations_2).shape[0],
        np.array(time).shape[0],
    )
    mpc_iter_durations_1 = mpc_iter_durations_1[:length]
    mpc_iter_durations_2 = mpc_iter_durations_2[:length]
    time = time[:length]

    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(title)
    ax.plot(time, mpc_iter_durations_1, label=label1)
    ax.plot(time, mpc_iter_durations_2, label=label2)
    ax.legend()
    ax.set_xlabel("t (s)")
    ax.set_ylabel("MPC iteration duration (s)")


def plot_xyz_traj(title, time, translation_pose_data, translation_pose_data_ref=None):
    fig, ax = plt.subplots(3, 1)
    fig.canvas.manager.set_window_title(title)
    axes = ["x", "y", "z"]
    for i in range(3):
        ax[i].plot(time, translation_pose_data[:, i], label=axes[i])
        if translation_pose_data_ref is not None:
            ax[i].plot(time, translation_pose_data_ref[:, i], label=axes[i] + " ref")
        ax[i].legend()
        ax[i].set_xlabel("t (s)")
        ax[i].set_ylabel(axes[i])


def plot_values(title, values, time, labels=None):
    values = np.array(values)
    fig, ax = plt.subplots(values.shape[1], 1)
    fig.canvas.manager.set_window_title(title)
    if values.shape[1] == 1:
        if labels is not None:
            ax.plot(time, values[:, 0], label=labels[0])
        else:
            ax.plot(time, values[:, i])
        ax.legend()
        ax.set_xlabel("t (s)")
    else:
        for i in range(values.shape[1]):
            if labels is not None:
                ax[i].plot(time, values[:, i], label=labels[i])
            else:
                ax[i].plot(time, values[:, i])
            ax[i].legend()
            ax[i].set_xlabel("t (s)")


def plot_values_on_same_fig(title, values, time, labels=None):
    values = np.array(values)
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(title)
    for i in range(values.shape[1]):
        if labels is not None:
            ax.semilogy(time, values[:, i], label=labels[i])
        else:
            ax.semilogy(time, values[:, i])
        ax.legend()
        ax.set_xlabel("t (s)")
