import time
import numpy as np
import yaml
import os
import pinocchio as pin
from copy import deepcopy
from mpc_utils.read_bags_utils import retrieve_data
from agimus_controller.utils.ocp_analyzer import (
    return_cost_vectors,
    return_constraint_vector,
    plot_costs_from_dic,
    plot_constraints_from_dic,
)

from agimus_controller_ros.agimus_controller import AgimusControllerNode
from agimus_controller_ros.parameters import AgimusControllerNodeParameters
from agimus_controller_ros.controller_base import HPPStateMachine
from agimus_controller.main.servers import Servers


class MPCReplay:
    def __init__(self, mpc_params, bag_path):
        self.bag_path = bag_path

        self.mpc_node = AgimusControllerNode(mpc_params)
        # self.mpc_data = {}

    def fill_buffer_offline(self):
        poses, _ = retrieve_data(self.bag_path, "/hpp/target/position", ["data"])
        vels, _ = retrieve_data(self.bag_path, "/hpp/target/velocity", ["data"])
        accs, _ = retrieve_data(self.bag_path, "/hpp/target/acceleration", ["data"])
        self.mpc_node.hpp_subscriber.fill_buffer_manually(poses, vels, accs)
        for _ in range(2 * self.mpc_node.params.ocp.horizon_size):
            self.mpc_node.fill_buffer()


def get_se3_from_ros_pose(position, orientation):
    pose_array = [
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]
    return pin.XYZQUATToSE3(pose_array)


def get_solver_models_ref_and_weight(solver):
    last_model = solver.problem.runningModels[-1]
    terminal_model = solver.problem.terminalModel
    last_model_grip_cost = last_model.differential.costs.costs["gripperPose"]
    terminal_model_grip_cost = terminal_model.differential.costs.costs["gripperPose"]
    print("last model ref ", last_model_grip_cost.cost.residual.reference)
    print("last model weight ", last_model_grip_cost.weight)
    print("terminal model ref ", terminal_model_grip_cost.cost.residual.reference)
    print("terminal model weight ", terminal_model_grip_cost.weight)

def save_hpp_traj(bag_path):
    poses, _ = retrieve_data(bag_path, "/hpp/target/position", ["data"])
    vels, _ = retrieve_data(bag_path, "/hpp/target/velocity", ["data"])
    accs, _ = retrieve_data(bag_path, "/hpp/target/acceleration", ["data"])
    new_poses = np.zeros((poses.shape[0],poses[0][0].shape[0]))
    new_vels = np.zeros((vels.shape[0],vels[0][0].shape[0]))
    new_accs = np.zeros((accs.shape[0],accs[0][0].shape[0]))
    for idx in range(new_poses.shape[0]):
        new_poses[idx,:] = poses[idx][0]
        new_vels[idx,:] = vels[idx][0]
        new_accs[idx,:] = accs[idx][0]
    dict = {}
    dict["poses"] = new_poses
    dict["vels"] = new_vels
    dict["accs"] = new_accs
    np.save("hpp_trajectory.npy",dict)


if __name__ == "__main__":
    with open("mpc_config.yaml", "r") as file:
        mpc_config = yaml.safe_load(file)

    bag_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])
    mpc_params_dict = np.load("mpc_params.npy", allow_pickle=True).item()
    mpc_params = AgimusControllerNodeParameters()
    mpc_params.set_parameters_from_dict(mpc_params_dict)
    mpc_params.activate_callback = False
    mpc_params.save_predictions_and_refs = True
    # mpc_params.horizon_size = 30
    # mpc_params.max_qp_iter = 200
    x0s, _ = retrieve_data(bag_path, "/ctrl_mpc_linearized/ocp_x0", ["joint_state"])
    states, _ = retrieve_data(bag_path, "/ctrl_mpc_linearized/state", ["data"])

    if mpc_params.use_vision is True:

        poses, _ = retrieve_data(
            bag_path, "/ctrl_mpc_linearized/happypose_pose", ["position"]
        )
        orientations, _ = retrieve_data(
            bag_path, "/ctrl_mpc_linearized/happypose_pose", ["orientation"]
        )
        length = min(
            states.shape[0], x0s.shape[0], poses.shape[0], orientations.shape[0]
        )
        poses = poses[:length]
        orientations = orientations[:length]
        mpc_params.use_vision = False
        mpc_replay = MPCReplay(mpc_params, bag_path)
        mpc_params.use_vision = True
        mpc_replay.mpc_node.params.use_vision = True
    else:
        length = min(states.shape[0], x0s.shape[0])
        mpc_replay = MPCReplay(mpc_params, bag_path)
    x0s = x0s[:length]
    states = states[:length]
    mpc_replay.mpc_node.state_machine = HPPStateMachine(states[0][0])

    x0 = np.concatenate([x0s[0][0].position, x0s[0][0].velocity])
    mpc_replay.fill_buffer_offline()
    start_solve_time = time.time()
    mpc_replay.mpc_node.first_solve(x0)
    solve_time = time.time() - start_solve_time
    mpc_replay.mpc_node.mpc.mpc_data["solve_time"] = [solve_time]
    if mpc_params.use_vision:
        mpc_replay.mpc_node.init_in_world_M_object = get_se3_from_ros_pose(
            poses[0][0], orientations[0][0]
        )
    elif mpc_params.use_vision_simulated:
        mpc_replay.mpc_node.simulate_happypose_callback()
    mpc_replay.mpc_node
    for idx in range(1, length):
        start_solve_time = time.time()
        mpc_replay.mpc_node.state_machine = HPPStateMachine(states[idx][0])
        x0 = np.concatenate([x0s[idx][0].position, x0s[idx][0].velocity])
        if mpc_params.use_vision:
            mpc_replay.mpc_node.in_world_M_object = get_se3_from_ros_pose(
                poses[idx][0], orientations[idx][0]
            )
        mpc_replay.mpc_node.solve(x0)

        # print(
        #    f"idx {idx} state {HPPStateMachine(states[idx][0])} val {mpc_replay.mpc_node.pick_traj_last_point_is_near(x0)} "
        # )
        if idx >= 516:

            print(
                "idx ",
                idx,
                " solver ",
                get_solver_models_ref_and_weight(mpc_replay.mpc_node.mpc.ocp.solver),
            )
        solve_time = time.time() - start_solve_time
        mpc_replay.mpc_node.mpc.mpc_data["solve_time"].append(solve_time)
        """
        if idx == 516:
            solver_516 = mpc_replay.mpc_node.mpc.ocp.solver
            idx_100_costs = return_cost_vectors(
                mpc_replay.mpc_node.mpc.ocp.solver, weighted=True
            )
            idx_100_constraint = return_constraint_vector(
                mpc_replay.mpc_node.mpc.ocp.solver
            )"""
        # if idx >= 516:
        #    breakpoint()
        # time.sleep(max(mpc_params.dt - solve_time, 0))
    np.save("mpc_data_replayed.npy", mpc_replay.mpc_node.mpc.mpc_data)
    save_hpp_traj(bag_path)


def compare_cost(cost_key, cost_dic, next_cost_dic):
    dic = {}
    dic[cost_key] = cost_dic[cost_key][1:]
    dic["new " + cost_key] = next_cost_dic[cost_key][:-1]
    plot_costs_from_dic(dic)


def compare_constraint(constraint_key, constraint_dic, next_constraint_dic):
    dic = {}
    dic[constraint_key] = constraint_dic[constraint_key][1:]
    dic["new " + constraint_key] = next_constraint_dic[constraint_key][:-1]
    plot_constraints_from_dic(dic)


