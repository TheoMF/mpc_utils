import os
import numpy as np
import yaml

import os
import numpy as np
import yaml
import pickle
from pathlib import Path
import xacro
from ament_index_python import get_package_share_directory
from agimus_controller_examples.utils.read_from_bag_trajectory import (
    save_rosbag_outputs_to_pickle,
)

from mpc_utils.plots_utils import plot_mpc_data

# from agimus_controller_examples.visualization.plots import MPCPlots
from agimus_controller_examples.utils.set_models_and_mpc import get_panda_models
from agimus_controller.factory.robot_model import RobotModelParameters, RobotModels


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


with open("mpc_config.yaml", "r") as file:
    mpc_config = yaml.safe_load(file)

bag_file_path = os.path.join(mpc_config["bag_directory"], mpc_config["bag_name"])
picke_file_path = mpc_config["bag_directory"] + "pickle_" + mpc_config["bag_name"]
save_rosbag_outputs_to_pickle(bag_file_path, picke_file_path)
with open(picke_file_path, "rb") as pickle_file:
    mpc_data = pickle.load(pickle_file)

if mpc_config["robot_name"] == "panda":
    config_folder_path = (
        Path(get_package_share_directory("agimus_demo_03_mpc_dummy_traj")) / "config"
    )
    env_xacro_path = (
        Path(get_package_share_directory("agimus_demo_05_pick_and_place"))
        / "urdf"
        / "environment.urdf.xacro"
    )
    robot_models = get_panda_models(config_folder_path, env_xacro_path)
elif mpc_config["robot_name"] == "tiago_pro":
    tiago_pro_pkg = Path(get_package_share_directory("tiago_pro_description"))
    tiago_pro_urdf_path = tiago_pro_pkg / "robots" / "tiago_pro.urdf.xacro"
    tiago_pro_urdf = xacro.process_file(tiago_pro_urdf_path).toxml()
    moving_joint_names = [
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_left_5_joint",
        "arm_left_6_joint",
        "arm_left_7_joint",
    ]
    robot_models_params = RobotModelParameters(
        robot_urdf=tiago_pro_urdf, moving_joint_names=moving_joint_names
    )
    robot_models = RobotModels(robot_models_params)
else:
    raise RuntimeError(f"Unknown robot name " + mpc_config["robot_name"])
rmodel = robot_models.robot_model


which_plots = [
    "computation_time",
    "collision_distance",
    "iter",
    "visual_servoing",
    "predictions",
]
plot_mpc_data(mpc_data, mpc_config, rmodel, which_plots)
