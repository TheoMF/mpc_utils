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
solve_time, time = retrieve_duration_data(
    bag_path, mpc_config["mpc_solve_time_topic_name"]
)
# plot_mpc_iter_durations("MPC iterations duration", solve_time, time)


print("solve time mean ", np.mean(solve_time))


mpc_data_replayed = np.load(
    mpc_config["bag_directory"] + "/mpc_data_replayed.npy", allow_pickle=True
).item()
"""
plot_2_mpc_iter_durations(
    "MPC iterations duration",
    solve_time,
    "online",
    mpc_data_replayed["solve_time"][: solve_time.shape[0]],
    "offline",
    time,
)
plt.show()"""


mpc_xs = np.array(mpc_data_replayed["preds_xs"])
mpc_us = np.array(mpc_data_replayed["preds_us"])
ctrl_refs = np.array(mpc_data_replayed["control_refs"])
state_refs = np.array(mpc_data_replayed["state_refs"])
translation_refs = np.array(mpc_data_replayed["translation_refs"])

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
