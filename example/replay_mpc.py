import time
import numpy as np
import yaml
import os
from mpc_utils.read_bags_utils import retrieve_data
from agimus_controller.utils.ocp_analyzer import (
    return_cost_vectors,
    return_constraint_vector,
    plot_costs_from_dic,
    plot_constraints_from_dic,
)

from agimus_controller_ros.agimus_controller import (
    AgimusControllerNode,
    AgimusControllerNodeParameters,
)
from agimus_controller_ros.controller_base import HPPStateMachine
from agimus_controller.main.servers import Servers


class MPCReplay:
    def __init__(self, mpc_params, bag_path):
        self.bag_path = bag_path
        self.mpc_node = AgimusControllerNode(mpc_params)
        self.mpc_data = {}

    def fill_buffer_offline(self):
        poses, _ = retrieve_data(self.bag_path, "/hpp/target/position", ["data"])
        vels, _ = retrieve_data(self.bag_path, "/hpp/target/velocity", ["data"])
        accs, _ = retrieve_data(self.bag_path, "/hpp/target/acceleration", ["data"])
        self.mpc_node.hpp_subscriber.fill_buffer_manually(poses, vels, accs)
        for _ in range(300):  # 2 * mpc_params.horizon_size
            self.mpc_node.fill_buffer()

    def create_mpc_data(self, solve_time):
        xs, us = self.mpc_node.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc_node.mpc.get_reference()

        self.mpc_data["preds_xs"] = [xs]
        self.mpc_data["preds_us"] = [us]
        self.mpc_data["state_refs"] = [x_ref]
        self.mpc_data["translation_refs"] = [p_ref]
        self.mpc_data["control_refs"] = [u_ref]
        self.mpc_data["solve_time"] = [solve_time]

    def fill_mpc_data(self, solve_time):
        xs, us = self.mpc_node.mpc.get_predictions()
        x_ref, p_ref, u_ref = self.mpc_node.mpc.get_reference()
        self.mpc_data["preds_xs"].append(xs)
        self.mpc_data["preds_us"].append(us)
        self.mpc_data["state_refs"].append(x_ref)
        self.mpc_data["translation_refs"].append(p_ref)
        self.mpc_data["control_refs"].append(u_ref)
        self.mpc_data["solve_time"].append(solve_time)


if __name__ == "__main__":
    with open("mpc_config.yaml", "r") as file:
        mpc_config = yaml.safe_load(file)

    bag_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])
    mpc_params_dict = np.load("mpc_params.npy", allow_pickle=True).item()
    mpc_params = AgimusControllerNodeParameters(False, mpc_params_dict)
    mpc_params.activate_callback = False
    mpc_params.save_predictions_and_refs = True
    x0s, _ = retrieve_data(bag_path, "/ctrl_mpc_linearized/ocp_x0", ["joint_state"])
    states, _ = retrieve_data(bag_path, "/ctrl_mpc_linearized/state", ["data"])
    length = min(states.shape[0], x0s.shape[0])
    x0s = x0s[:length]
    states = states[:length]
    mpc_replay = MPCReplay(mpc_params, bag_path)
    mpc_replay.mpc_node.state_machine = HPPStateMachine(states[0][0])

    x0 = np.concatenate([x0s[0][0].position, x0s[0][0].velocity])
    mpc_replay.fill_buffer_offline()
    start_solve_time = time.time()
    mpc_replay.mpc_node.first_solve(x0)
    solve_time = time.time() - start_solve_time
    mpc_replay.mpc_node.mpc_data["solve_time"] = [solve_time]
    mpc_replay.mpc_node

    for idx in range(1, x0s.shape[0]):
        mpc_replay.mpc_node.state_machine = HPPStateMachine(states[idx][0])
        x0 = np.concatenate([x0s[idx][0].position, x0s[idx][0].velocity])
        start_solve_time = time.time()
        mpc_replay.mpc_node.solve(x0)
        print(
            f"idx {idx} state {HPPStateMachine(states[idx][0])} val {mpc_replay.mpc_node.pick_traj_last_point_is_near(x0)} "
        )
        solve_time = time.time() - start_solve_time
        mpc_replay.mpc_node.mpc_data["solve_time"].append(solve_time)

        if idx == 100:
            idx_100_costs = return_cost_vectors(
                mpc_replay.mpc_node.mpc.ocp.solver, weighted=True
            )
            idx_100_constraint = return_constraint_vector(
                mpc_replay.mpc_node.mpc.ocp.solver
            )
        elif idx == 101:
            idx_101_costs = return_cost_vectors(
                mpc_replay.mpc_node.mpc.ocp.solver, weighted=True
            )
            idx_101_constraint = return_constraint_vector(
                mpc_replay.mpc_node.mpc.ocp.solver
            )
        elif idx == 4:
            idx_4_costs = return_cost_vectors(
                mpc_replay.mpc_node.mpc.ocp.solver, weighted=True
            )
            idx_4_constraint = return_constraint_vector(
                mpc_replay.mpc_node.mpc.ocp.solver
            )

    np.save("mpc_data_replayed.npy", mpc_replay.mpc_node.mpc_data)


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