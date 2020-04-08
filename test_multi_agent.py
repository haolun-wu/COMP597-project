from __future__ import absolute_import, division, print_function

import argparse
import os

from collections import defaultdict
from math import log

from test_product_agent import evaluate_product_paths, predict_product_paths
from test_agent import evaluate_paths, predict_paths
from utils import *


def evaluate_multi(topk_user_matches, topk_product_matches, test_labels, num_recommendations, re_rank, brand_dict):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    # create a new topk_matches by cross-referencing the results of both agents
    topk_matches = defaultdict(list)
    for uid, pids in topk_user_matches.items():
        for pid in pids:
            if len(topk_matches[uid]) >= num_recommendations:
                break
            try:
                if uid in topk_product_matches[pid]:
                    topk_matches[uid].append(pid)
            except KeyError:
                continue
    # fill in the rest of the recommendation set with matches from the user set (so effectively
    # you're just changing the order of recommendations - i.e re-ranking)
    if re_rank:
        for uid, pids in topk_user_matches.items():
            if len(topk_matches[uid]) < num_recommendations:
                for pid in pids:
                    if pid not in topk_matches[uid]:
                        topk_matches[uid].append(pid)

    # Compute metrics
    invalid_users = []
    precisions, recalls, ndcgs, hits, fairness = [], [], [], [], []
    test_user_idxs = list(test_labels.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < num_recommendations:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_labels[uid]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)
        fairness.append(calculate_fairness(pred_list, brand_dict))

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    avg_fairness = np.mean(fairness)
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Fairness={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, avg_fairness, len(invalid_users)))


def test(args):
    user_policy_file = args.log_dir_user + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    user_path_file = args.log_dir_user + '/policy_paths_epoch{}.pkl'.format(args.epochs)
    product_policy_file = args.log_dir_product + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    product_path_file = args.log_dir_product + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')

    if args.run_path:
        predict_paths(user_policy_file, user_path_file, args)
        predict_product_paths(product_policy_file, product_path_file, args)
    if args.run_eval:
        pred_user_labels = evaluate_paths(user_path_file, train_labels, test_labels, args.num_recommendations * args.base_rec_multiplier, args)
        pred_product_labels = evaluate_product_paths(product_path_file, train_labels, test_labels, args.num_recommendations * args.base_rec_multiplier, args)
        evaluate_multi(pred_user_labels, pred_product_labels, test_labels, args.num_recommendations, args.re_rank, args.brand_dict)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='multi_agent', help='directory name.')
    parser.add_argument('--name_user_agent', type=str, default='train_agent', help='directory name where the user '
                                                                                   'agent stored it\'s files.')
    parser.add_argument('--name_product_agent', type=str, default='train_product_agent', help='directory name where the'
                                                                                              ' product agent stored'
                                                                                              ' it\'s files.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_users', type=boolean, default=False, help='add predicted users up to num_recommendations')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=False, help='Generate predicted paths? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--num_recommendations', type=int, default=10, help='The number of recommendations that '
                                                                             'will be predicted for each product')
    parser.add_argument('--base_rec_multiplier', type=int, default=3, help='Control how many more k '
                                                                                  'user/products the base agents will '
                                                                                  'suggest')
    parser.add_argument('--re_rank', type=boolean, default=False, help='Attempt to fill in user recommendations to'
                                                                       ' reach the number of recommendations, effectively'
                                                                       ' doing a re_ranking')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    args.log_dir_user = TMP_DIR[args.dataset] + '/' + args.name_user_agent
    args.log_dir_product = TMP_DIR[args.dataset] + '/' + args.name_product_agent

    pickle_in = open(BRAND_FILE[args.dataset], "rb")
    args.brand_dict = pickle.load(pickle_in)

    test(args)

