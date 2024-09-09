import matplotlib.pyplot as plt


def plot_mpc_iter_durations(title, mpc_iter_durations, time):
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title(title)
    ax.plot(time, mpc_iter_durations)
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
