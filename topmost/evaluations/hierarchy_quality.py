import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer

from .build_hierarchy import build_hierarchy


def parse_item_info(topic_str):
    item_info = topic_str.split()[0]
    layer_id, topic_idx = (int(item.split('-')[1]) for item in item_info.split('_'))
    return layer_id, topic_idx


def convert_topicStr_to_dict(topic_str_list):
    """

    Args:
        topic_str_list: [L-0_K-0 w1 w2 w3 ..., L-0_K-1 w1 w2 w3 ...]. L indicates the layer, and K indicates the topic.
        keep_info (bool, optional): if keep the item info L-0_K-0. Defaults to False.

    Returns:
        hierarchical_topic_dict: {0: ["w1 w2...", "w1 w2..."], 1: ["w1 w2...", "w1 w2..."]}

    """

    hierarchical_topic_dict = defaultdict(list)

    for topic_str in topic_str_list:
        topic_str_items = topic_str.split()
        layer_id, k = parse_item_info(topic_str)

        dict_item = ' '.join(topic_str_items)

        hierarchical_topic_dict[layer_id].append(dict_item)

    return hierarchical_topic_dict


def compute_TD(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer()
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD


def compute_CLNPMI(parent_diff_words, child_diff_words, all_bow, vocab):
    npmi_list = list()

    for p_w in parent_diff_words:
        flag_n = all_bow[:, vocab.index(p_w)] > 0
        p_n = np.sum(flag_n) / len(all_bow)

        for c_w in child_diff_words:
            flag_l = all_bow[:, vocab.index(c_w)] > 0
            p_l = np.sum(flag_l)
            p_nl = np.sum(flag_n * flag_l)

            if p_nl == len(all_bow):
                npmi_score = 1
            else:
                p_l = p_l / len(all_bow)
                p_nl = p_nl / len(all_bow)
                p_nl += 1e-10
                npmi_score = np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)

            npmi_list.append(npmi_score)

    return npmi_list


def get_CLNPMI(PC_pair_groups, all_bow, vocab):
    CNPMI_list = list()
    for group in tqdm(PC_pair_groups):
        layer_CNPMI = list()
        for parent_topic, child_topic in group:
            parent_words = set(parent_topic.split())
            child_words = set(child_topic.split())

            inter = parent_words.intersection(child_words)
            parent_diff_words = list(parent_words.difference(inter))
            child_diff_words = list(child_words.difference(inter))

            npmi_list = compute_CLNPMI(parent_diff_words, child_diff_words, all_bow, vocab)

            # NOTE: assign -1 to the NPMI of repetitive word pairs
            num_repetition = (len(parent_words) - len(parent_diff_words)) * (len(child_words) - len(child_diff_words))
            npmi_list.extend([-1] * num_repetition)

            layer_CNPMI.extend(npmi_list)

        CNPMI_list.append(np.mean(layer_CNPMI))

    return CNPMI_list


def compute_diff_topic_pair(topic_str_a, topic_str_b):
    word_counter = Counter()
    topic_words_a = topic_str_a.split()
    topic_words_b = topic_str_b.split()
    word_counter.update(topic_words_a)
    word_counter.update(topic_words_b)
    diff = (np.asarray(list(word_counter.values())) == 1).sum() / (len(topic_words_a) + len(topic_words_b))
    return diff


def get_topics_difference(topic_pair_groups):
    diff_list = list()
    for groups in topic_pair_groups:
        layer_diff = list()
        for topic_a, topic_b in groups:
            diff = compute_diff_topic_pair(topic_a, topic_b)
            layer_diff.append(diff)
        diff_list.append(np.mean(layer_diff))

    return diff_list


# Given a list of child topics, find the nonchild topics based on their item info.
def extract_nonchild_topic_list(hierarchical_topic_dict, child_topic_list, num_topics_list):
    child_topic_idx_list = [parse_item_info(child_topic)[1] for child_topic in child_topic_list]
    layer_id, _ = parse_item_info(child_topic_list[0])
    # num_topic = beta_list[layer_id].shape[0]
    num_topic = num_topics_list[layer_id]
    nonchild_topic_idx_list = list(set(range(num_topic)) - set(child_topic_idx_list))
    nonchild_topic_list = np.asarray(hierarchical_topic_dict[layer_id])[nonchild_topic_idx_list].tolist()

    return nonchild_topic_list


def get_topic_pairs(topic_pairs, topic_hierarchy, hierarchical_topic_dict, num_topics_list, _type, layer_id=0):
    for parent_topic in topic_hierarchy.keys():
        if isinstance(topic_hierarchy[parent_topic], list):
            child_topic_list = topic_hierarchy[parent_topic]
        else:
            child_topic_list = list(topic_hierarchy[parent_topic].keys())

        if _type == 'PC':
            for child_topic in child_topic_list:
                topic_pairs[layer_id].append((parent_topic, child_topic))

        elif _type == 'PnonC':
            nonchild_topic_list = extract_nonchild_topic_list(hierarchical_topic_dict, child_topic_list, num_topics_list)
            for nonchild_topic in nonchild_topic_list:
                topic_pairs[layer_id].append((parent_topic, nonchild_topic))

        # Move to the next layer if more.
        if not isinstance(topic_hierarchy[parent_topic], list):
            get_topic_pairs(topic_pairs, topic_hierarchy[parent_topic], hierarchical_topic_dict, num_topics_list, _type, layer_id + 1)


# sibling_groups: length == num_layers
# each element in the list is a group of sibling topics at a layer.
def get_sibling_groups(topic_hierarchy, sibling_groups, layer_id=0):
    if isinstance(topic_hierarchy, list):
        sibling_groups[layer_id].append(topic_hierarchy)
    else:
        # sibling topics at this layer
        sibling_groups[layer_id].append(list(topic_hierarchy.keys()))
        # sibling topics at next layer
        for parent_topic in topic_hierarchy.keys():
            get_sibling_groups(topic_hierarchy[parent_topic], sibling_groups, layer_id + 1)


def get_Sibling_TD(sibling_groups):
    sibling_TD = list()
    for group in sibling_groups:
        layer_sibling_TD = list()
        for sibling_topics in group:
            TD = compute_TD(sibling_topics)
            layer_sibling_TD.append(TD)
        sibling_TD.append(np.mean(layer_sibling_TD))
    return sibling_TD


def get_Sibling_NPMI(sibling_groups, all_bow, vocab):
    sibling_NPMI = list()
    for group in sibling_groups:
        layer_pairs = list()
        for sibling_topics in group:
            sibling_num = len(sibling_topics)
            for i in range(sibling_num):
                for j in range(i + 1, sibling_num):
                    layer_pairs.append([sibling_topics[i], sibling_topics[j]])

        npmi = get_CLNPMI(layer_pairs, all_bow, vocab)
        sibling_NPMI.append(np.mean(npmi))
    return sibling_NPMI


def get_topic_groups(hierarchical_topic_dict, topic_hierarchy, beta_list):
    num_layers = len(beta_list)
    num_topics_list = [item.shape[0] for item in beta_list]

    PC_pair_groups = [list() for _ in range(num_layers - 1)]
    PnonC_pair_groups = [list() for _ in range(num_layers - 1)]

    get_topic_pairs(PC_pair_groups, topic_hierarchy, hierarchical_topic_dict, num_topics_list, _type='PC')
    get_topic_pairs(PnonC_pair_groups, topic_hierarchy, hierarchical_topic_dict, num_topics_list, _type='PnonC')

    sibling_groups = [list() for _ in range(num_layers)]
    get_sibling_groups(topic_hierarchy, sibling_groups)

    # Because hierarhcial_topic_dict contains item info of each topic (Layer-0_K-20)
    # Remove these item info from topic strings
    PC_pair_groups = clean_group_info(PC_pair_groups)
    PnonC_pair_groups = clean_group_info(PnonC_pair_groups)
    sibling_groups = clean_group_info(sibling_groups)
    return PC_pair_groups, PnonC_pair_groups, sibling_groups


def clean_group_info(groups):
    clean_groups = list()
    for layer_group in groups:
        layer_clean_group = list()
        for topic_str_list in layer_group:
            layer_clean_group.append(clean_info(topic_str_list))
        clean_groups.append(layer_clean_group)
    return clean_groups


# remove item info of topic strings
# L-0_K-0 w1 w2 ===> w1 w2.
def clean_info(topic_str_list):
    clean_list = [" ".join(item.split()[1:]) for item in topic_str_list]
    return clean_list


def hierarchy_quality(vocab, reference_bow, topic_str_list, beta_list, phi_list):
    hierarchical_topic_dict = convert_topicStr_to_dict(topic_str_list)
    topic_hierarchy = build_hierarchy(hierarchical_topic_dict, phi_list)

    PC_pair_groups, PnonC_pair_groups, sibling_groups = get_topic_groups(hierarchical_topic_dict, topic_hierarchy, beta_list)

    # Parent and Child topic Coherence (PCC)
    CLNPMI = get_CLNPMI(PC_pair_groups, reference_bow, vocab)

    # Parent and Child topic Diversity (PCD)
    PC_TD = get_topics_difference(PC_pair_groups)

    # Sibling Topic Diversity (SD)
    Sibling_TD = get_Sibling_TD(sibling_groups)

    # Parent and non-Child Topic Diversity (PnCD)
    PnonC_TD = get_topics_difference(PnonC_pair_groups)

    rst = {
        'PCC': np.mean(CLNPMI),
        'PCD': np.mean(PC_TD),
        'Sibling_TD': np.mean(Sibling_TD),
        'PnCD': np.mean(PnonC_TD)
    }

    return rst, topic_hierarchy
