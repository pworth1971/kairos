
##########################################################################################
# Evaluation Script for Provenance-based Intrusion Detection (CADETS_E3)
#
# Purpose:
#   - Compare detected anomalous time windows against ground-truth attack intervals
#   - Compute performance metrics: Precision, Recall, F1, Accuracy, AUC
#   - Log confusion matrix elements and per-metric scores
#
# Workflow:
#   1. Load anomaly "history lists" (graph_4_6, graph_4_7)
#   2. Calculate anomaly scores per queue
#   3. Mark time windows as predicted attacks if anomaly score > β threshold
#   4. Compare with known attack intervals (ground_truth_label)
#   5. Output detailed metrics + summary
##########################################################################################



from sklearn.metrics import confusion_matrix
import logging

from kairos_utils import *
from model import *


# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + 'evaluation.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# --------------------------------------------------------------------------
# Helper: Evaluate classifier metrics
# --------------------------------------------------------------------------
def classifier_evaluation(y_test, y_test_pred):
    """
    Compute confusion matrix and derived metrics.

    Args:
        y_test (list[int]): Ground truth labels (0 = benign, 1 = attack)
        y_test_pred (list[int]): Predicted labels from the model

    Returns:
        precision, recall, fscore, accuracy, auc_val
    """

    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()

    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)

    # Log detailed stats
    logger.info(f"Confusion Matrix → TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    logger.info(f"Precision:{precision:.4f} Recall:{recall:.4f} F1:{fscore:.4f} Accuracy:{accuracy:.4f} AUC:{auc_val:.4f}")

    # Print summary to console
    print("\n📊 Evaluation Metrics Summary:")
    print(f"  True Negatives : {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives : {tp}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {fscore:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUC:       {auc_val:.4f}\n")

    return precision,recall,fscore,accuracy,auc_val


# --------------------------------------------------------------------------
# Ground truth: known attack intervals for CADETS E3
# --------------------------------------------------------------------------
def ground_truth_label():
    """
    Create a mapping of time-window filenames → labels.
    Windows that correspond to known attack periods are labeled 1.
    """

    labels = {}
    filelist = os.listdir(f"{artifact_dir}/graph_4_6")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_7")
    for f in filelist:
        labels[f] = 0

    attack_list = [
        '2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
        '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
        '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
        '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
    ]
    for i in attack_list:
        labels[i] = 1

    return labels


# --------------------------------------------------------------------------
# Helper: Optional diagnostic — count number of attack edges by IP/keyword
# --------------------------------------------------------------------------
def calc_attack_edges():
    """
    Searches anomalous edge logs for known malicious indicators (IP, process names).
    Used to sanity-check reconstruction accuracy.
    """

    def keyword_hit(line):
        attack_nodes = [
            'vUgefal',
            '/var/log/devc',
            'nginx',
            '81.49.200.166',
            '78.205.235.65',
            '200.36.109.214',
            '139.123.0.113',
            '152.111.159.139',
            '61.167.39.128',

        ]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    attack_list = [
        '2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
        '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
        '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
        '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
    ]
    for f in attack_list:
        files.append(f"{artifact_dir}/graph_4_6/{f}")

    attack_edge_count = 0
    for fpath in (files):
        f = open(fpath)
        for line in f:
            if keyword_hit(line):
                attack_edge_count += 1

    logger.info(f"Detected attack-related edges: {attack_edge_count}")
    print(f"Detected attack-related edges: {attack_edge_count}")


# --------------------------------------------------------------------------
# Main Evaluation Execution
# --------------------------------------------------------------------------
if __name__ == "__main__":

    logger.info("Start logging.")

    # Step 1. Validation set → find maximum observed anomaly score (Validation date)
    anomalous_queue_scores = []                                         
    history_list = torch.load(f"{artifact_dir}/graph_4_5_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                # Plus 1 to ensure anomaly score is monotonically increasing
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []

        for i in hl:
            name_list.append(i['name'])
        # logger.info(f"Constructed queue: {name_list}")
        # logger.info(f"Anomaly score: {anomaly_score}")

        anomalous_queue_scores.append(anomaly_score)

    max_val_score = max(anomalous_queue_scores)
    logger.info(f"Max anomaly score in validation: {max_val_score:.4f}")
    print(f"\n[Validation] Max Anomaly Score: {max_val_score:.4f}")

    # Step 2. Evaluate testing set (graph_4_6 & graph_4_7)
    pred_label = {}

    filelist = os.listdir(f"{artifact_dir}/graph_4_6/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_4_7/")
    for f in filelist:
        pred_label[f] = 0

    # Graph 4_6 and 4_7 - apply detection threshold
    history_list = torch.load(f"{artifact_dir}/graph_4_6_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day6:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{artifact_dir}/graph_4_7_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day7:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")

    # Step 3. Evaluate performance vs ground truth
    labels = ground_truth_label()
    y = []
    y_pred = []
    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label[i])

    precision, recall, fscore, accuracy, auc_val = classifier_evaluation(y, y_pred)

    # Step 4. Optional diagnostic for edge-level indicators
    calc_attack_edges()

    # Step 5. Print summary to console
    print("\n================ Evaluation Summary ================")
    print(f"Validation Max Anomaly Score: {max_val_score:.4f}")
    print(f"Test Set Precision : {precision:.4f}")
    print(f"Test Set Recall    : {recall:.4f}")
    print(f"Test Set F1-Score  : {fscore:.4f}")
    print(f"Test Set Accuracy  : {accuracy:.4f}")
    print(f"Test Set AUC       : {auc_val:.4f}")
    print("===================================================")

    logger.info("Evaluation complete.")
