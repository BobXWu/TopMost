import numpy as np


def get_child_topic_idx(phi, topic_idx):
    num_topic, next_num_topic = phi.shape
    num_child_topic = round(next_num_topic / num_topic)

    row = phi[topic_idx]
    child_topic_idx_list = np.argsort(row)[:-(num_child_topic + 1):-1].tolist()

    return child_topic_idx_list


def build_hierarchy(hierarchical_topic_dict, phi_list, topic_idx_list=None, layer_id=0):
    # first layer.
    if topic_idx_list is None:
        topic_idx_list = list(range(phi_list[0].shape[0]))

    # last layer, where layer_id == L-1.
    if layer_id >= len(phi_list):
        # return the topic strings at the last layer.
        hierarchy = np.asarray(hierarchical_topic_dict[layer_id])[topic_idx_list].tolist()
        return hierarchy

    # NOT the last layer.
    hierarchy = dict()
    phi = phi_list[layer_id]

    for topic_idx in topic_idx_list:
        child_topic_idx_list = get_child_topic_idx(phi, topic_idx)
        hierarchy[hierarchical_topic_dict[layer_id][topic_idx]] = build_hierarchy(hierarchical_topic_dict, phi_list, child_topic_idx_list, layer_id + 1)

    return hierarchy
