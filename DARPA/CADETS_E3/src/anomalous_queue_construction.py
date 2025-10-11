import logging

import torch

from kairos_utils import *
from config import *

# Setting for logging
logger = logging.getLogger("anomalous_queue_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'anomalous_queue.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


##########################################################################################
# Anomalous Queue Construction
#
# This script post-processes the temporal reconstruction losses produced by the model
# to identify time windows containing clusters of anomalous activity.
#
# It computes:
#   - Per-node "inverse document frequency" (IDF) scores to weigh node rarity
#   - Per-time-window anomaly metrics (based on edge reconstruction losses)
#   - Cross-window relationships (via shared rare nodes) to form temporal "queues"
#     of related anomalies, reflecting long-term multi-step attack behavior.
#
# Outputs:
#   - node_IDF: per-node rarity weights (saved to artifact_dir)
#   - *_history_list: serialized anomaly queue structures for visualization and analysis
#
# Typical workflow:
#   1. Model outputs per-edge reconstruction losses per day (in graph_4_5, graph_4_6, etc.)
#   2. This script aggregates those results and links related anomalies across days.
##########################################################################################


# --------------------------------------------------------------------------
# Function: cal_anomaly_loss
# Purpose: Identify anomalous edges within a single time window
# --------------------------------------------------------------------------
def cal_anomaly_loss(loss_list, edge_list):
    """
    Given per-edge reconstruction losses, determine which edges are anomalous.

    Args:
        loss_list (list[float]): list of per-edge reconstruction losses
        edge_list (list[list[str]]): corresponding source/dest node pairs

    Returns:
        count (int): number of anomalous edges
        avg_loss (float): mean loss of anomalous edges
        node_set (set): nodes participating in anomalies
        edge_set (set): unique anomalous edges
    """

    if len(loss_list) != len(edge_list):
        print("error!")
        return 0
    
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    edge_set = set()
    node_set = set()

    # Define anomaly threshold as mean + 1.5 * stddev
    thr = loss_mean + 1.5 * loss_std
    logger.info(f"Anomaly threshold (mean + 1.5σ): {thr}")

    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1
            src_node = edge_list[i][0]
            dst_node = edge_list[i][1]
            loss_sum += loss_list[i]

            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0] + edge_list[i][1])
    
    # Return metrics for this time window
    return count, loss_sum / count, node_set, edge_set



# --------------------------------------------------------------------------
# Function: compute_IDF
# Purpose: Compute per-node rarity weights across all observed time windows
# --------------------------------------------------------------------------
def compute_IDF():
    """
    Computes an Inverse Document Frequency (IDF) score for each node:
    - Nodes that appear in fewer time windows get higher IDF (rarer).
    - Common system nodes (e.g., /proc, /usr) are filtered or down-weighted.

    Returns:
        node_IDF (dict): mapping node string → rarity weight
        file_list (list): list of processed time window files
    """

    #
    # Collect all anomaly result files across multiple days
    # 
    node_IDF = {}
    file_list = []
    file_path = artifact_dir + "graph_4_3/"
    file_l = os.listdir(file_path)
    
    for i in file_l:
        file_list.append(file_path + i)

    file_path = artifact_dir + "graph_4_4/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = artifact_dir + "graph_4_5/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    # Build node occurrence sets
    node_set = {}
    for f_path in tqdm(file_list):
        f = open(f_path)
        for line in f:
            l = line.strip()
            jdata = eval(l)
            if jdata['loss'] > 0:
                if 'netflow' not in str(jdata['srcmsg']):
                    if str(jdata['srcmsg']) not in node_set.keys():
                        node_set[str(jdata['srcmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['srcmsg'])].add(f_path)
                if 'netflow' not in str(jdata['dstmsg']):
                    if str(jdata['dstmsg']) not in node_set.keys():
                        node_set[str(jdata['dstmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['dstmsg'])].add(f_path)
    
    # Compute IDF: log(total_files / (1 + count_in_files))
    for n in node_set:
        include_count = len(node_set[n])
        IDF = math.log(len(file_list) / (include_count + 1))
        node_IDF[n] = IDF

    torch.save(node_IDF, artifact_dir + "node_IDF")
    logger.info("IDF weight calculate complete!")
    return node_IDF, file_list


# --------------------------------------------------------------------------
# Function: cal_set_rel
# Purpose: Quantify overlap between anomalous node sets in two time windows
# Method: Measure the relationship between two time windows, if the returned value
# is not 0, it means there are suspicious nodes in both time windows.
# --------------------------------------------------------------------------
def cal_set_rel(s1, s2, node_IDF, tw_list):
    """
    Measures whether two time windows share rare anomalous nodes.
    This helps link anomalies across time (building the "queue").

    Returns:
        count (int): number of rare overlapping nodes between windows
    """

    def is_include_key_word(s):
        """
        Filters out common or uninformative system nodes (noise).
        """
        # The following common nodes don't exist in the training/validation data, but
        # will have the influences to the construction of anomalous queue (i.e. noise).
        # These nodes frequently exist in the testing data but don't contribute much to
        # the detection (including temporary files or files with random name).
        # Assume the IDF can keep being updated with the new time windows, these
        # common nodes can be filtered out.
        keywords = [
            'netflow',
            '/home/george/Drafts',
            'usr',
            'proc',
            'var',
            'cadet',
            '/var/log/debug.log',
            '/var/log/cron',
            '/home/charles/Drafts',
            '/etc/ssl/cert.pem',
            '/tmp/.31.3022e',
        ]
        flag = False
        for i in keywords:
            if i in s:
                flag = True
        return flag

    new_s = s1 & s2
    count = 0
    for i in new_s:
        if is_include_key_word(i) is True:
            node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))

        if i in node_IDF.keys():
            IDF = node_IDF[i]
        else:
            # Assign a high IDF for those nodes which are neither in training/validation
            # sets nor excluded node list above.
            IDF = math.log(len(tw_list) / (1))

        # Compare the IDF with a rareness threshold α
        if IDF > (math.log(len(tw_list) * 0.9)):
            logger.info(f"node:{i}, IDF:{IDF}")
            count += 1
    return count

# --------------------------------------------------------------------------
# Function: anomalous_queue_construction
# Purpose: Build sequences of related anomalies (queues) across time windows
# --------------------------------------------------------------------------
def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path):
    """
    Iterates through anomaly logs for each day/time window, computes anomalous edges,
    and connects temporally adjacent windows sharing rare nodes.
    """

    history_list = []
    current_tw = {}

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):
        logger.info("**************************************************")
        logger.info(f"Time window: {f_path}")

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []
        logger.info(f'Time window index: {index_count}')

        # Figure out which nodes are anomalous in this time window
        for line in f:
            l = line.strip()
            jdata = eval(l)
            edge_loss_list.append(jdata['loss'])
            edge_list.append([str(jdata['srcmsg']), str(jdata['dstmsg'])])
        count, loss_avg, node_set, edge_set = cal_anomaly_loss(edge_loss_list, edge_list)
        current_tw['name'] = f_path
        current_tw['loss'] = loss_avg
        current_tw['index'] = index_count
        current_tw['nodeset'] = node_set

        # Incrementally construct the queues
        added_que_flag = False
        for hq in history_list:
            for his_tw in hq:
                if cal_set_rel(current_tw['nodeset'], his_tw['nodeset'], node_IDF, tw_list) != 0 and current_tw['name'] != his_tw['name']:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
                if added_que_flag:
                    break
        if added_que_flag is False:
            temp_hq = [copy.deepcopy(current_tw)]
            history_list.append(temp_hq)

        index_count += 1

        # Log summary metrics for this time window
        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list)}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list


# --------------------------------------------------------------------------
# Main Execution: Compute anomaly queues for validation and test datasets
# --------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Anomalous Queue Construction...")

    node_IDF, tw_list = compute_IDF()

    # Validation date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_5/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_5_history_list")

    # Testing date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_6/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_6_history_list")

    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_7/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_7_history_list")