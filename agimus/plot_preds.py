import os
import numpy as np
import yaml
import warnings

import os
import numpy as np
import yaml
import pickle
from agimus_controller_examples.utils.read_from_bag_trajectory import (
    save_rosbag_outputs_to_pickle,
)

from mpc_utils.plots_utils import (
    plot_mpc_iter_durations,
    plot_values,
    concatenate_array_with_list_of_arrays,
)
from mpc_utils.plot_tails import plot_tails

# from agimus_controller_examples.visualization.plots import MPCPlots
from agimus_controller_examples.utils.set_models_and_mpc import get_panda_models


def get_time_per_iteration(nb_iters, step_times):
    time_per_iters = step_times.copy()
    for idx, val in enumerate(nb_iters):
        if val > 0:
            time_per_iters[idx] /= val
    return time_per_iters


def get_nb_in_saturation_constraint(col_values, safety_margin, eps):
    nb_iteration = col_values.shape[0]
    nb_constraints = col_values.shape[1]
    res = np.zeros((nb_iteration))
    for idx in range(nb_iteration):
        nb_cons_in_saturation = 0
        for cons_idx in range(nb_constraints):
            if col_values[idx, cons_idx] - safety_margin < eps:
                nb_cons_in_saturation += 1
        res[idx] = nb_cons_in_saturation
    return res


robot_models = get_panda_models("agimus_demo_03_mpc_dummy_traj")
rmodel = robot_models.robot_model
cmodel = robot_models.collision_model
vmodel = robot_models.visual_model
with open("mpc_config.yaml", "r") as file:
    mpc_config = yaml.safe_load(file)


bag_file_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])
picke_file_path = mpc_config["bag_directory"] + "pickle_" + mpc_config["bag_name"]
save_rosbag_outputs_to_pickle(bag_file_path, picke_file_path)
with open(picke_file_path, "rb") as pickle_file:
    mpc_data = pickle.load(pickle_file)


mpc_xs = np.array(mpc_data["states_predictions"])
mpc_us = np.array(mpc_data["control_predictions"])
ctrl_refs = np.array(mpc_data["control_reg_references"])
state_refs = np.array(mpc_data["state_reg_references"])
translation_refs = np.array(mpc_data["goal_tracking_references"])[:, :3]
if "distance" in mpc_data.keys():
    coll_distance_residuals = np.array(mpc_data["distance"])
solve_time = np.array(mpc_data["solve_time"])


# PLOT COMPUTATION TIME
time = np.linspace(0, (solve_time.shape[0] - 1) * 0.01, solve_time.shape[0])
plot_mpc_iter_durations("MPC iterations duration", solve_time, time)
print("solve time mean ", np.mean(solve_time))

# PLOT COLLISIONS DISTANCE
if "distance" in mpc_data.keys():
    coll_distance_residuals = coll_distance_residuals[:, 0, np.newaxis]
    time_col = np.linspace(
        0,
        (coll_distance_residuals.shape[0] - 1) * 0.01,
        coll_distance_residuals.shape[0],
    )
    coll_labels = [f"col_term_{i}" for i in range(coll_distance_residuals.shape[0])]
    plot_values(
        "collision pairs distances",
        coll_distance_residuals,
        time_col,
        coll_labels,
        # ylimits=[[0.01, 0.06]] * 9,
    )
# except Exception:
#    warnings.warn("no collisions residuals or not it correct format")


# PLOT KKT / ITERATIONS

try:
    kkt_norms = np.array(mpc_data["kkt_norms"])
    nb_iters = np.array(mpc_data["nb_iters"])
    nb_qp_iters = np.array(mpc_data["nb_qp_iters"])
    per_iter_avg_times = get_time_per_iteration(
        nb_iters=nb_iters, step_times=solve_time
    )
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
except KeyError:
    warnings.warn("data not in correct format")


# MPC Plots for Meshcat viewer
"""
mpc_plots = MPCPlots(
    croco_xs=mpc_xs[:, 0, :],
    croco_us=mpc_us[:, 0, :],
    whole_x_plan=mpc_xs[:, 0, :],
    whole_u_plan=mpc_us[:, 0, :],
    rmodel=rmodel,
    vmodel=vmodel,
    cmodel=cmodel,
    DT=0.01,
    ee_frame_name=mpc_config["endeff_name"],
    viewer=None,
)
"""
# Plot predictions
plot_tails(
    mpc_xs,
    mpc_us,
    rmodel,
    mpc_config,
    ctrl_refs=ctrl_refs,
    state_refs=state_refs,
    translation_refs=translation_refs,
)
