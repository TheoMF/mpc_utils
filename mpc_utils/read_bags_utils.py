import numpy as np

from pathlib import Path

from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader


def reshape_mpc_data(df_mpc, nx, nv, nb_running_nodes):
    data_mpc = df_mpc.data.to_numpy()
    nb_mpc_iter = data_mpc.shape[0]
    mpc_xs = np.zeros([nb_mpc_iter, nb_running_nodes + 1, nx])
    mpc_us = np.zeros([nb_mpc_iter, nb_running_nodes, nv])
    for mpc_iter in range(nb_mpc_iter):
        for idx in range(nb_running_nodes):
            mpc_xs[mpc_iter, idx, :] = data_mpc[mpc_iter][
                idx * (nx + nv) : idx * (nx + nv) + nx
            ]
            mpc_us[mpc_iter, idx, :] = data_mpc[mpc_iter][
                idx * (nx + nv) + nx : (idx + 1) * (nx + nv)
            ]
        mpc_xs[mpc_iter, nb_running_nodes, :] = data_mpc[mpc_iter][
            (nb_running_nodes - 1) * (nx + nv) : (nb_running_nodes - 1) * (nx + nv) + nx
        ]
    return mpc_xs, mpc_us


def retrieve_mpc_data(bag_path, topic, nx, nv, nb_running_nodes):
    with AnyReader([Path(bag_path)]) as reader:
        df_mpc = get_dataframe(reader, topic, ["data"])
    return reshape_mpc_data(df_mpc, nx, nv, nb_running_nodes)


def get_bag_topic_time(index):
    # Compute time indices
    idx_arr = index.to_numpy()
    t_delta_arr = idx_arr
    ns2sec = np.vectorize(lambda x: float(x) / 1e9)
    return ns2sec(t_delta_arr)


def convert_ros_duration_to_numpy(data):
    converted_data = np.zeros(data.shape)
    for idx, val in enumerate(data):
        converted_data[idx] = val.sec + val.nanosec / 1e9
    return converted_data


def retrieve_duration_data(bag_path, topic):
    with AnyReader([Path(bag_path)]) as reader:
        df = get_dataframe(reader, topic, ["data"])
    duration_array = convert_ros_duration_to_numpy(df.data.to_numpy())
    return duration_array, get_bag_topic_time(df.index)


def retrieve_data(bag_path, topic, keys):
    with AnyReader([Path(bag_path)]) as reader:
        df = get_dataframe(reader, topic, keys)
        return df.to_numpy(), get_bag_topic_time(df.index)


def retrieve_topic_time(bag_path, topic, field):
    with AnyReader([Path(bag_path)]) as reader:
        df = get_dataframe(reader, topic, [field])
    return get_bag_topic_time(df.index)
