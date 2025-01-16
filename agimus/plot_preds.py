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
from agimus_controller.visualization.plots import MPCPlots
from agimus_controller.robot_model.panda_model import get_task_models


def get_lin_pos_ee_pred(
    mpc_xs, mpc_us, model, mpc_config, ctrl_refs, state_refs, translation_refs
):
    sim_data, _ = get_sim_data(
        mpc_xs, mpc_us, model, mpc_config, ctrl_refs, state_refs, translation_refs
    )
    plot_data = extract_plot_data_from_sim_data(sim_data)
    return plot_data["lin_pos_ee_pred"]


def concatenate_arrays_columns(array1, array2):
    if len(array1.shape) == len(array2.shape):
        array1 = array1[:, np.newaxis]
    return np.c_[array1[: array2.shape[0]], array2[:, np.newaxis]]



def get_np_array_from_ros_msg_3d_array(datas):
    res = np.zeros((datas.shape[0], 3))
    for i in range(datas.shape[0]):
        res[i, 0] = datas[i][0].x
        res[i, 1] = datas[i][0].y
        res[i, 2] = datas[i][0].z
    return res


def get_time_per_iteration(nb_iters, step_times):
    time_per_iters = step_times.copy()
    for idx, val in enumerate(nb_iters):
        if val > 0:
            time_per_iters[idx] /= val
    return time_per_iters

def get_nb_in_saturation_constraint(col_values,eps):
    nb_iteration = col_values.shape[0]
    nb_constraints = col_values.shape[1]
    res = np.zeros((nb_iteration))
    for idx in range(nb_iteration):
        nb_cons_in_saturation = 0
        for cons_idx in range(nb_constraints):
            if col_values[idx,cons_idx]-eps < 1e-3:
                nb_cons_in_saturation +=1
        res[idx] = nb_cons_in_saturation
    return res



rmodel, cmodel, vmodel = get_task_models(task_name="pick_and_place")
with open("mpc_config.yaml", "r") as file:
    mpc_config = yaml.safe_load(file)

bag_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])


mpc_data = np.load(
    mpc_config["bag_directory"] + "/mpc_data.npy",
    allow_pickle=True,
).item()

mpc_xs = np.array(mpc_data["preds_xs"])
mpc_us = np.array(mpc_data["preds_us"])
ctrl_refs = np.array(mpc_data["control_refs"])
state_refs = np.array(mpc_data["state_refs"])
translation_refs = np.array(mpc_data["translation_refs"])
mpc_data_2pairs = np.load(
    mpc_config["bag_directory"] + "/mpc_data_2collision_pairs_30_iter.npy",
    allow_pickle=True,
).item()
mpc_data_3pairs = np.load(
    mpc_config["bag_directory"] + "/mpc_data_3collision_pairs_30_iter.npy",
    allow_pickle=True,
).item()
# PLOT COMPUTATION TIME
solve_time, time = retrieve_duration_data(
    bag_path, mpc_config["mpc_solve_time_topic_name"]
)
#solve_time = np.array(mpc_data["step_time"])
time = np.linspace(0, (solve_time.shape[0] - 1) * 0.01, solve_time.shape[0])
plot_mpc_iter_durations("MPC iterations duration", solve_time, time)
print("solve time mean ", np.mean(solve_time))

# PLOT COLLISIONS DISTANCE
col_values = np.array(mpc_data["coll_residuals"]["col_term_0"])
col_values = col_values[:, np.newaxis]
for key in mpc_data["coll_residuals"].keys():
    if key == "col_term_0":
        continue
    new_col_values = np.array(mpc_data["coll_residuals"][key])
    col_values = concatenate_arrays_columns(col_values, new_col_values)
time_col = np.linspace(0, (col_values.shape[0] - 1) * 0.01, col_values.shape[0])
plot_values(
    "collision pairs distances",
    col_values,
    time_col,
    [" link5 obs1"," link7 obs1","leftfinger obs1", " link3 obs2"," link5 obs2","leftfinger obs2"," link3 obs3"," link5 obs3","leftfinger obs3"],
    ylimits=[[0.01,0.06]]*9
)
#["leftfinger obs1", "link5 obs2", " leftfinger obs2", " link5 obs1", " leftfinger obs3"]


# PLOT KKT / ITERATIONS
kkt_iters_values = np.array(mpc_data["kkt_norm"])
nb_iters = np.array(mpc_data["nb_iter"])
#nb_qp_iters = np.array(mpc_data["nb_qp_iter"])
#step_times = np.array(mpc_data["step_time"])
#per_iter_avg_times = get_time_per_iteration(nb_iters=nb_iters, step_times=step_times)
kkt_iters_values = concatenate_arrays_columns(kkt_iters_values, nb_iters)
#kkt_iters_values = concatenate_arrays_columns(kkt_iters_values, nb_qp_iters)
#kkt_iters_values = concatenate_arrays_columns(kkt_iters_values, step_times)
#kkt_iters_values = concatenate_arrays_columns(kkt_iters_values, per_iter_avg_times)
nb_saturated_cons = get_nb_in_saturation_constraint(col_values,eps=0.02)
kkt_iters_values = concatenate_arrays_columns(kkt_iters_values, nb_saturated_cons)
time_kkt = np.linspace(0, (kkt_iters_values.shape[0] - 1) * 0.01, kkt_iters_values.shape[0])

plot_values(
    "kkt norm",
    kkt_iters_values,
    time_kkt[: kkt_iters_values.shape[0]],
    ["kkt termination tol 1e-4 ", "nb iterations", "nb saturated cons", "total_solve_times", "per_iter_avg_times"],
    semilogs=[True, True, False, False, False],
)

# Compare time steps depending on nb of collision pairs

#per_iter_avg_times = get_time_per_iteration(nb_iters=nb_iters, step_times=step_times)
#step_times_2pairs = np.array(mpc_data_2pairs["step_time"])
#step_times_3pairs = np.array(mpc_data_3pairs["step_time"])
#per_iter_times_2pairs = get_time_per_iteration(
#    nb_iters=np.array(mpc_data_2pairs["nb_iter"]), step_times=step_times_2pairs
#)
#per_iter_times_3pairs = get_time_per_iteration(
#    nb_iters=np.array(mpc_data_3pairs["nb_iter"]), step_times=step_times_3pairs
#)
#per_iter_avg_times = concatenate_arrays_columns(per_iter_avg_times, per_iter_times_2pairs)
#per_iter_avg_times = concatenate_arrays_columns(per_iter_avg_times, per_iter_times_3pairs)
#plot_values_on_same_fig(
#    "per iter step times",
#    per_iter_avg_times,
#    time_col[: per_iter_avg_times.shape[0]],
#    [
#        "per iter step times 1 pair of collision",
#        "per iter step times 2 pairs of collision",
#        "per iter step times 3 pairs of collision",
#    ],
#)

# Compare time steps depending on nb of collision pairs

#step_times = np.array(mpc_data["step_time"])
#
#step_times = concatenate_arrays_columns(step_times, step_times_2pairs)
#step_times = concatenate_arrays_columns(step_times, step_times_3pairs)
#plot_values_on_same_fig(
#    "step times",
#    step_times,
#    time_col[: step_times.shape[0]],
#    [
#        "step times 1 pair of collision",
#        "step times 2 pairs of collision",
#        "step times 3 pairs of collision",
#    ],
#)


# Plot end_effector trajectory
#pos_pred = get_lin_pos_ee_pred(
#    mpc_xs, mpc_us, rmodel, mpc_config, ctrl_refs, state_refs, translation_refs
#)
#time = np.linspace(0, (pos_pred.shape[0] - 1) * 0.01, pos_pred.shape[0])
#plot_xyz_traj("ee pose ", time, pos_pred[:, 0, :])

# Plot Vision
#try:
#    ros_poses, time = retrieve_data(
#        bag_path, "/ctrl_mpc_linearized/happypose_pose", ["position"]
#    )
#    poses = get_np_array_from_ros_msg_3d_array(ros_poses)
#    plot_xyz_traj("vision pose ", time, poses)
#except:
#    print("no happypose detections logged in bag file")

# Plot force
try:
    force_raw_data, force_time = retrieve_data(bag_path, "/sensor_force", ["vector"])
    force_values = get_np_array_from_ros_msg_3d_array(force_raw_data)
    force_time -= force_time[0]
    plot_values(
        "force sensor noise ",
        force_values[:200],
        force_time[:200],
        ["x", "y", "z"],
        ylabels="Force (N)",
    )
except:
    print("no force data logged in bag file")


# MPC Plots for Meshcat viewer
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
