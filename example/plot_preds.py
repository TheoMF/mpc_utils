import os
import numpy as np
import yaml
from mpc_utils.mpc_utils import extract_plot_data_from_sim_data
from mpc_utils.read_bags_utils import retrieve_duration_data
from mpc_utils.plots import (
    plot_mpc_iter_durations,
    plot_xyz_traj,
    plot_values,
    plot_values_on_same_fig,
)
from mpc_utils.read_bags_utils import retrieve_data
from mpc_utils.plot_tails import plot_tails, get_sim_data
from agimus_controller.robot_model.panda_model import PandaRobotModel


def get_lin_pos_ee_pred(
    mpc_xs, mpc_us, model, mpc_config, ctrl_refs, state_refs, translation_refs
):
    sim_data, _ = get_sim_data(
        mpc_xs, mpc_us, model, mpc_config, ctrl_refs, state_refs, translation_refs
    )
    plot_data = extract_plot_data_from_sim_data(sim_data)
    return plot_data["lin_pos_ee_pred"]


def concatenate_arrays_columns(array1, array2):
    array1 = array1[:, np.newaxis]
    return np.c_[array1[: array2.shape[0]], array2[:, np.newaxis]]


def get_np_array_from_ros_msg_3d_array(datas):
    res = np.zeros((datas.shape[0], 3))
    for i in range(datas.shape[0]):
        res[i, 0] = datas[i][0].x
        res[i, 1] = datas[i][0].y
        res[i, 2] = datas[i][0].z
    return res


robot_constructor = PandaRobotModel.load_model()
robot_model = robot_constructor.get_reduced_robot_model()
with open("mpc_config.yaml", "r") as file:
    mpc_config = yaml.safe_load(file)

bag_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])


mpc_data = np.load(
    mpc_config["bag_directory"] + "/mpc_data_send_to_theo.npy",
    allow_pickle=True,
).item()

mpc_xs = np.array(mpc_data["preds_xs"])
mpc_us = np.array(mpc_data["preds_us"])
ctrl_refs = np.array(mpc_data["control_refs"])
state_refs = np.array(mpc_data["state_refs"])
translation_refs = np.array(mpc_data["translation_refs"])
mpc_data_replayed = np.load(
    mpc_config["bag_directory"] + "/mpc_data_replayed.npy", allow_pickle=True
).item()


# PLOT COMPUTATION TIME
solve_time, time = retrieve_duration_data(
    bag_path, mpc_config["mpc_solve_time_topic_name"]
)
plot_mpc_iter_durations("MPC iterations duration", solve_time, time)
print("solve time mean ", np.mean(solve_time))

# PLOT COLLISIONS DISTANCE
col_values = np.array(mpc_data["coll_residuals"]["col_term_0"])
col_values_1 = np.array(mpc_data["coll_residuals"]["col_term_1"])
col_values = concatenate_arrays_columns(col_values, col_values_1)
time_col = np.linspace(0, (col_values.shape[0] - 1) * 0.01, col_values.shape[0])
plot_values(
    "collision pairs distances",
    col_values,
    time_col,
    ["leftfinger pair", "rightfinger pair"],
)


# PLOT KKT
kkt_values = np.array(mpc_data["kkt_norm"])
time_kkt = np.linspace(0, (kkt_values.shape[0] - 1) * 0.01, kkt_values.shape[0])
plot_values_on_same_fig(
    "kkt norm",
    kkt_values[:, np.newaxis],
    time_col[: kkt_values.shape[0]],
    ["kkt 100 iter, termination tol 1e-4 ", "kkt 100 iter, termination tol 1e-4"],
)


# Plot end_effector trajectory
pos_pred = get_lin_pos_ee_pred(
    mpc_xs, mpc_us, robot_model, mpc_config, ctrl_refs, state_refs, translation_refs
)
time = np.linspace(0, (pos_pred.shape[0] - 1) * 0.01, pos_pred.shape[0])
plot_xyz_traj("ee pose ", time, pos_pred[:, 0, :])

# Plot Vision
ros_poses, time = retrieve_data(
    bag_path, "/ctrl_mpc_linearized/happypose_pose", ["position"]
)
poses = get_np_array_from_ros_msg_3d_array(ros_poses)
plot_xyz_traj("vision pose ", time, poses)


# Plot force
try:
    force_raw_data, force_time = retrieve_data(bag_path, "/sensor_force", ["vector"])
    force_values = get_np_array_from_ros_msg_3d_array(force_raw_data)
    force_time -= force_time[0]
    plot_values(
        "force sensor noise ",
        force_values,
        force_time,
        ["x", "y", "z"],
        ylabel="Force (N)",
    )
except:
    print("no force data logged in bag file")


# Plot predictions
plot_tails(
    mpc_xs,
    mpc_us,
    robot_model,
    mpc_config,
    ctrl_refs=ctrl_refs,
    state_refs=state_refs,
    translation_refs=translation_refs,
)
