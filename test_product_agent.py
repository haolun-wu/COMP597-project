"""Run this file to train the User Agent"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import itertools
from functools import reduce

from math import log
from tqdm import tqdm

from kg_env_product import BatchKGProductEnvironment
from train_product_agent import ActorCriticProduct
from utils import *


def evaluate_product(topk_matches, test_product_users, num_recommendations):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
        test_product_users: the data to test against
        num_recommendations: the value of k in the topk
    """
    invalid_products = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_product_idxs = list(test_product_users.keys())
    for pid in test_product_idxs:
        if pid not in topk_matches or len(topk_matches[pid]) < num_recommendations:
            invalid_products.append(pid)
            continue
        pred_list, rel_set = topk_matches[pid][::-1], test_product_users[pid]
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

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid products={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_products)))


def batch_beam_product_search(env, model, pids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(pids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in pids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_product_paths(policy_file, path_file, args):
    print('Predicting paths...')
    env = BatchKGProductEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = torch.load(policy_file)
    model = ActorCriticProduct(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.dataset, 'test')
    test_pids = list(itertools.chain(*test_labels.values()))

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_pids))
    while start_idx < len(test_pids):
        end_idx = min(start_idx + batch_size, len(test_pids))
        batch_pids = test_pids[start_idx:end_idx]
        paths, probs = batch_beam_product_search(env, model, batch_pids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_product_paths(path_file, train_labels, test_labels, num_recommendations, args):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    purchase_embeds = embeds[PURCHASE][0]
    product_embeds = embeds[PRODUCT]
    scores = np.dot(product_embeds, (user_embeds + purchase_embeds).T)

    # 1) Get all valid paths for each product, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {pid: {} for pid in list(itertools.chain(*test_labels.values()))}
    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != USER:
            continue
        pid = path[0][2]
        if pid not in pred_paths:
            continue
        uid = path[-1][2]
        if uid not in pred_paths[pid]:
            pred_paths[pid][uid] = []
        path_score = scores[pid][uid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[pid][uid].append((path_score, path_prob, path))

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    inv_train_labels = invert_labels(train_labels)
    for pid in pred_paths:
        train_uids = inv_train_labels[pid]
        best_pred_paths[pid] = []
        for uid in pred_paths[pid]:
            if uid in train_uids:
                continue
            # Get the path with highest probability
            sorted_path = sorted(pred_paths[pid][uid], key=lambda x: x[1], reverse=True)
            best_pred_paths[pid].append(sorted_path[0])

    # 3) Compute top k recommended products for each user.
    sort_by = 'score'
    pred_labels = {}
    for pid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[pid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[pid], key=lambda x: (x[1], x[0]), reverse=True)
        top_k_uids = [p[-1][2] for _, _, p in sorted_path[:num_recommendations]]  # from largest to smallest
        # add up to 10 pids if not enough
        if args.add_users and len(top_k_uids) < num_recommendations:
            train_uids = inv_train_labels[pid]
            cand_uids = np.argsort(scores[pid])
            for cand_uid in cand_uids[::-1]:
                if cand_uid in train_uids or cand_uid in top_k_uids:
                    continue
                top_k_uids.append(cand_uid)
                if len(top_k_uids) >= num_recommendations:
                    break
        # end of add
        pred_labels[pid] = top_k_uids[::-1]  # change order to from smallest to largest!

    return pred_labels


def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')

    if args.run_path:
        predict_product_paths(policy_file, path_file, args)
    if args.run_eval:
        pred_labels = evaluate_product_paths(path_file, train_labels, test_labels, args.num_recommendations, args)
        inv_test_labels = invert_labels(test_labels)
        evaluate_product(pred_labels, inv_test_labels, args.num_recommendations)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_product_agent', help='directory name.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_users', type=boolean, default=False, help='add predicted users up to num_recommendations')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--num_recommendations', type=int, default=10, help='The number of recommendations that '
                                                                            'will be predicted for each product')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    test(args)

