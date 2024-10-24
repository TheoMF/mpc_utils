import os

import numpy as np
import example_robot_data
import pinocchio as pin
import yaml
import matplotlib.pyplot as plt
from mpc_utils.mpc_utils import extract_plot_data_from_sim_data
from mpc_utils.read_bags_utils import retrieve_duration_data, get_bag_topic_time
from mpc_utils.plots import (
    plot_mpc_iter_durations,
    plot_xyz_traj,
    plot_2_mpc_iter_durations,
    plot_values,
    plot_values_on_same_fig,
)
from mpc_utils.plot_tails import plot_tails, get_sim_data
from agimus_controller.robot_model.panda_model import PandaRobotModel


def get_robot_model(robot):
    locked_joints = [
        robot.model.getJointId("panda_finger_joint1"),
        robot.model.getJointId("panda_finger_joint2"),
    ]

    urdf_path = "robot.urdf"
    srdf_path = "demo.srdf"

    model = pin.Model()
    pin.buildModelFromUrdf(urdf_path, model)
    pin.loadReferenceConfigurations(model, srdf_path, False)
    q0 = model.referenceConfigurations["default"]
    return pin.buildReducedModel(model, locked_joints, q0)


robot_constructor = PandaRobotModel.load_model()
robot = example_robot_data.load("panda")
model = robot_constructor.get_reduced_robot_model()
robot.model = model
with open("mpc_config.yaml", "r") as file:
    mpc_config = yaml.safe_load(file)

bag_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])


mpc_data = np.load(
    mpc_config["bag_directory"] + "/mpc_data_1000_it_ter_1e-4_ugrav.npy",
    allow_pickle=True,
).item()
mpc_data_2 = np.load(
    mpc_config["bag_directory"] + "/mpc_data_100_it_ter_1e-4_ugrav.npy",
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

# plot_2_mpc_iter_durations(
#    "MPC iterations duration",
#    solve_time,
#    "online",
#    mpc_data_replayed["solve_time"][: solve_time.shape[0]],
#    "offline",
#    time,
# )

# PLOT COMPUTATION TIME
solve_time, time = retrieve_duration_data(
    bag_path, mpc_config["mpc_solve_time_topic_name"]
)
plot_mpc_iter_durations("MPC iterations duration", solve_time, time)
print("solve time mean ", np.mean(solve_time))

# PLOT COLLISIONS DISTANCE
col_values = np.array(mpc_data["coll_residuals"]["col_term_0"])
col_values = col_values[:, np.newaxis]
""""
col_values_1 = np.array(mpc_data["coll_residuals"]["col_term_1"])
# col_values_2 = np.array(mpc_data["coll_residuals"]["col_term_2"])
col_values = np.r_[col_values, col_values_1[np.newaxis, :]]"""
# col_values = np.r_[col_values, col_values_2[np.newaxis, :]]
time_col = np.linspace(0, (col_values.shape[0] - 1) * 0.01, col_values.shape[0])
print("shape ", col_values.shape)
plot_values(
    "collision pairs distances",
    col_values,
    time_col,
    ["leftfinger pair", "rightfinger pair", "pair 2"],
)

# PLOT KKT
kkt_values = np.array(mpc_data["kkt_norm"])
kkt_values = kkt_values[:, np.newaxis]
time_kkt = np.linspace(0, (kkt_values.shape[0] - 1) * 0.01, kkt_values.shape[0])
"""
kkt_values_1000_it = np.array(mpc_data_2["kkt_norm"])

kkt_values = np.c_[
    kkt_values[: kkt_values_1000_it.shape[0]], kkt_values_1000_it[:, np.newaxis]
]"""

plot_values_on_same_fig(
    "kkt norm",
    kkt_values,
    time_col[: kkt_values.shape[0]],
    ["kkt 1000 iter, termination tol 1e-4 ", "kkt 100 iter, termination tol 1e-4"],
)

# PLOT XYZ TRAJ
"""
sim_data_500_it, _ = get_sim_data(
    mpc_xs, mpc_us, model, mpc_config, ctrl_refs, state_refs, translation_refs
)
plot_data_500_it = extract_plot_data_from_sim_data(sim_data_500_it)
mpc_xs_1000_it = np.array(mpc_data_2["preds_xs"])
mpc_us_1000_it = np.array(mpc_data_2["preds_us"])
ctrl_refs_1000_it = np.array(mpc_data_2["control_refs"])
state_refs_1000_it = np.array(mpc_data_2["state_refs"])
translation_refs_1000_it = np.array(mpc_data_2["translation_refs"])
sim_data_1000_it, _ = get_sim_data(
    mpc_xs_1000_it,
    mpc_us_1000_it,
    model,
    mpc_config,
    ctrl_refs_1000_it,
    state_refs_1000_it,
    translation_refs_1000_it,
)
plot_data_1000_it = extract_plot_data_from_sim_data(sim_data_1000_it)
plot_values_on_same_fig(
    "x",
    np.concatenate(
        [
            plot_data_500_it["lin_pos_ee_pred"][:, 0, 0, np.newaxis],
            plot_data_1000_it["lin_pos_ee_pred"][:, 0, 0, np.newaxis],
        ],
        axis=1,
    ),
    time_kkt[:-1],
    ["x 500 iter", "x 1000 iter"],
)
plot_values_on_same_fig(
    "y",
    np.concatenate(
        [
            plot_data_500_it["lin_pos_ee_pred"][:, 0, 1, np.newaxis],
            plot_data_1000_it["lin_pos_ee_pred"][:, 0, 1, np.newaxis],
        ],
        axis=1,
    ),
    time_kkt[:-1],
    ["y 500 iter", "y 1000 iter"],
)
plot_values_on_same_fig(
    "z",
    np.concatenate(
        [
            plot_data_500_it["lin_pos_ee_pred"][:, 0, 2, np.newaxis],
            plot_data_1000_it["lin_pos_ee_pred"][:, 0, 2, np.newaxis],
        ],
        axis=1,
    ),
    time_kkt[:-1],
    ["z 500 iter", "z 1000 iter"],
)
"""
"""
mpc_xs = np.array(mpc_data_replayed["preds_xs"])
mpc_us = np.array(mpc_data_replayed["preds_us"])
ctrl_refs = np.array(mpc_data_replayed["control_refs"])
state_refs = np.array(mpc_data_replayed["state_refs"])
translation_refs = np.array(mpc_data_replayed["translation_refs"])
"""
time = np.linspace(0, (translation_refs.shape[0] - 1) * 0.01, translation_refs.shape[0])
"""
if "vision_refs" in mpc_data.keys():
    vision_data = np.array(mpc_data["vision_refs"])

    plot_xyz_traj("vision pose ", time, vision_data)


sim_data, sim_params = get_sim_data(
    mpc_xs, mpc_us, model, mpc_config, ctrl_refs, state_refs, translation_refs
)
plot_data = extract_plot_data_from_sim_data(sim_data)
last_point = plot_data["lin_pos_ee_pred"][-1, 1, :]

plot_xyz_traj(
    "ee pose ",
    time[:-1],
    np.concatenate((plot_data["lin_pos_ee_pred"][:, 0, :], last_point[np.newaxis, :])),
    translation_refs[:-1],
)

plt.show()
"""
plot_tails(
    mpc_xs,
    mpc_us,
    robot.model,
    mpc_config,
    ctrl_refs=ctrl_refs,
    state_refs=state_refs,
    translation_refs=translation_refs,
)
