
# import torch
# import random
# import json
# import numpy as np
# import pdb
# import torch.distributed as dist
# import os
# import os.path as osp
# from collections import Counter
# import pickle
# import torch.nn.functional as F
# from transformers import BertTokenizer
# import torch.distributed
# from tqdm import tqdm
# import networkx as nx

# from .utils import get_topk_indices, get_adjr


# class EADataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]


# class Collator_base(object):
#     def __init__(self, args):
#         self.args = args

#     def __call__(self, batch):
#         return np.array(batch)


# def calculate_betweenness_importance(adj, bc_sample_size=1000, seed=42):
#     """
#     基于近似介数中心性计算节点全局重要性。

#     参数：
#         adj: 由知识图谱三元组构造得到的稀疏邻接矩阵
#         bc_sample_size: 近似计算介数中心性时采样的节点数量
#         seed: 随机种子，保证近似计算可复现

#     返回：
#         scores: torch.Tensor, shape [num_nodes]
#     """
#     print("开始计算近似介数中心性...")

#     # 1. 从稀疏邻接矩阵中提取边
#     indices = adj._indices().cpu().numpy()
#     num_nodes = adj.size(0)

#     # 2. 构建无向图
#     G = nx.Graph()
#     G.add_nodes_from(range(num_nodes))
#     G.add_edges_from(indices.T)

#     # 3. 移除自环
#     G.remove_edges_from(nx.selfloop_edges(G))

#     # 4. 采样数量不能超过节点总数
#     bc_sample_size = min(bc_sample_size, num_nodes)

#     print(f"正在计算近似介数中心性，采样节点数: {bc_sample_size}/{num_nodes}")

#     # 5. 近似介数中心性
#     bc_dict = nx.betweenness_centrality(
#         G,
#         k=bc_sample_size,
#         normalized=True,
#         weight=None,
#         seed=seed
#     )

#     # 6. 转成 Tensor，保证顺序和实体 id 一致
#     scores = torch.zeros(num_nodes, dtype=torch.float32)
#     for node, score in bc_dict.items():
#         if node < num_nodes:
#             scores[node] = score

#     # 7. Min-Max 归一化
#     min_val = scores.min()
#     max_val = scores.max()

#     if max_val > min_val:
#         scores = (scores - min_val) / (max_val - min_val + 1e-9)
#     else:
#         scores = torch.ones_like(scores)

#     # 8. 平滑，避免大量 0 导致 beta 分支失效
#     scores = scores + 1e-3

#     print(
#         "介数中心性计算完成！",
#         f"min={scores.min().item():.6f},",
#         f"max={scores.max().item():.6f},",
#         f"mean={scores.mean().item():.6f}"
#     )

#     return scores


# def load_data(logger, args):
#     assert args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]

#     if args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]:
#         KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_eva_data(logger, args)
#     elif args.data_choice in ["FBYG15K_attr", "FBDB15K_attr"]:
#         pass

#     return KGs, non_train, train_ill, test_ill, eval_ill, test_ill_


# def load_eva_data(logger, args):
#     file_dir = osp.join(args.data_path, args.data_choice, args.data_split)
#     lang_list = [1, 2]

#     ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(file_dir, lang_list)

#     e1 = os.path.join(file_dir, 'ent_ids_1')
#     e2 = os.path.join(file_dir, 'ent_ids_2')

#     left_ents = get_ids(e1)
#     right_ents = get_ids(e2)

#     ENT_NUM = len(ent2id_dict)
#     REL_NUM = len(r_hs)

#     np.random.shuffle(ills)

#     if "V1" in file_dir:
#         split = "norm"
#         img_vec_path = osp.join(args.data_path, "pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl")
#     elif "V2" in file_dir:
#         split = "dense"
#         img_vec_path = osp.join(args.data_path, "pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl")
#     elif "FB" in file_dir:
#         img_vec_path = osp.join(args.data_path, f"pkls/{args.data_choice}_id_img_feature_dict.pkl")
#     else:
#         split = file_dir.split("/")[-1]
#         img_vec_path = osp.join(args.data_path, "pkls", args.data_split + "_GA_id_img_feature_dict.pkl")

#     assert osp.exists(img_vec_path)

#     img_features = load_img(logger, ENT_NUM, img_vec_path)
#     logger.info(f"image feature shape:{img_features.shape}")

#     if args.word_embedding == "glove":
#         word2vec_path = os.path.join(args.data_path, "embedding", "glove.6B.300d.txt")
#     elif args.word_embedding == "bert":
#         pass
#     else:
#         raise Exception("error word embedding")

#     name_features = None
#     char_features = None

#     if args.data_choice == "DBP15K" and (args.w_name or args.w_char):
#         assert osp.exists(word2vec_path)

#         ent_vec, char_features = load_word_char_features(ENT_NUM, word2vec_path, args, logger)

#         name_features = F.normalize(torch.Tensor(ent_vec))
#         char_features = F.normalize(torch.Tensor(char_features))

#         logger.info(f"name feature shape:{name_features.shape}")
#         logger.info(f"char feature shape:{char_features.shape}")

#     if args.unsup:
#         mode = args.unsup_mode

#         if mode == "char":
#             input_features = char_features
#         elif mode == "name":
#             input_features = name_features
#         else:
#             input_features = F.normalize(torch.Tensor(img_features))

#         train_ill = visual_pivot_induction(args, left_ents, right_ents, input_features, ills, logger)
#     else:
#         train_ill = np.array(ills[:int(len(ills) // 1 * args.data_rate)], dtype=np.int32)

#     test_ill_ = ills[int(len(ills) // 1 * args.data_rate):]
#     test_ill = np.array(test_ill_, dtype=np.int32)

#     test_left = torch.LongTensor(test_ill[:, 0].squeeze())
#     test_right = torch.LongTensor(test_ill[:, 1].squeeze())

#     left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))
#     right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

#     logger.info(f"#left entity : {len(left_ents)}, #right entity: {len(right_ents)}")
#     logger.info(
#         f"#left entity not in train set: {len(left_non_train)}, "
#         f"#right entity not in train set: {len(right_non_train)}"
#     )

#     rel_features = load_relation(ENT_NUM, triples, 1000)
#     logger.info(f"relation feature shape:{rel_features.shape}")

#     a1 = os.path.join(file_dir, 'training_attrs_1')
#     a2 = os.path.join(file_dir, 'training_attrs_2')

#     att_features, bow_dim = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)  # BoW + BERT attr
#     logger.info(f"attribute feature shape:{att_features.shape}")
#     logger.info(f"bow_dim:{bow_dim}")

#     logger.info("-----dataset summary-----")
#     logger.info(f"dataset:\t\t {file_dir}")
#     logger.info(f"triple num:\t {len(triples)}")
#     logger.info(f"entity num:\t {ENT_NUM}")
#     logger.info(f"relation num:\t {REL_NUM}")
#     logger.info(f"train ill num:\t {train_ill.shape[0]} \t test ill num:\t {test_ill.shape[0]}")
#     logger.info("-------------------------")

#     eval_ill = None

#     input_idx = torch.LongTensor(np.arange(ENT_NUM))
#     adj = get_adjr(ENT_NUM, triples, norm=True)

#     # ===== 新增：计算或加载介数中心性重要性分数 =====
#     bc_sample_size = getattr(args, "bc_sample_size", 1000)
#     random_seed = getattr(args, "random_seed", 42)

#     importance_path = osp.join(file_dir, f"betweenness_importance_bc{bc_sample_size}.pt")

#     if osp.exists(importance_path):
#         importance_scores = torch.load(importance_path, map_location="cpu")
#         logger.info(f"load betweenness importance from {importance_path}")
#     else:
#         logger.info("calculating approximate betweenness importance scores...")
#         importance_scores = calculate_betweenness_importance(
#             adj,
#             bc_sample_size=bc_sample_size,
#             seed=random_seed
#         )
#         torch.save(importance_scores.cpu(), importance_path)
#         logger.info(f"save betweenness importance to {importance_path}")

#     logger.info(
#         f"importance score shape: {importance_scores.shape}, "
#         f"min: {importance_scores.min().item():.6f}, "
#         f"max: {importance_scores.max().item():.6f}, "
#         f"mean: {importance_scores.mean().item():.6f}"
#     )

#     train_ill = EADataset(train_ill)
#     test_ill = EADataset(test_ill)

#     return {
#         'ent_num': ENT_NUM,
#         'rel_num': REL_NUM,
#         'images_list': img_features,
#         'rel_features': rel_features,
#         'att_features': att_features,
#         'bow_dim': bow_dim,
#         'name_features': name_features,
#         'char_features': char_features,
#         'input_idx': input_idx,
#         'adj': adj,
#         'importance_scores': importance_scores,
#     }, {"left": left_non_train, "right": right_non_train}, train_ill, test_ill, eval_ill, test_ill_


# def load_word2vec(path, dim=300):
#     word2vec = dict()
#     err_num = 0
#     err_list = []

#     with open(path, 'r', encoding='utf-8') as file:
#         for line in tqdm(file.readlines(), desc="load word embedding"):
#             line = line.strip('\n').split(' ')

#             if len(line) != dim + 1:
#                 continue

#             try:
#                 v = np.array(list(map(float, line[1:])), dtype=np.float64)
#                 word2vec[line[0].lower()] = v
#             except:
#                 err_num += 1
#                 err_list.append(line[0])
#                 continue

#     print("err list ", err_list)
#     print("err num ", err_num)

#     return word2vec


# def load_char_bigram(path):
#     ent_names = json.load(open(path, "r"))
#     char2id = {}
#     count = 0

#     for _, name in ent_names:
#         for word in name:
#             word = word.lower()

#             for idx in range(len(word) - 1):
#                 if word[idx:idx + 2] not in char2id:
#                     char2id[word[idx:idx + 2]] = count
#                     count += 1

#     return ent_names, char2id


# def load_word_char_features(node_size, word2vec_path, args, logger):
#     name_path = os.path.join(args.data_path, "DBP15K", "translated_ent_name", "dbp_" + args.data_split + ".json")

#     assert osp.exists(name_path)

#     save_path_name = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_name.pkl")
#     save_path_char = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_char.pkl")

#     if osp.exists(save_path_name) and osp.exists(save_path_char):
#         logger.info(f"load entity name emb from {save_path_name} ... ")
#         ent_vec = pickle.load(open(save_path_name, "rb"))

#         logger.info(f"load entity char emb from {save_path_char} ... ")
#         char_vec = pickle.load(open(save_path_char, "rb"))

#         return ent_vec, char_vec

#     word_vecs = load_word2vec(word2vec_path)
#     ent_names, char2id = load_char_bigram(name_path)

#     ent_vec = np.zeros((node_size, 300))
#     char_vec = np.zeros((node_size, len(char2id)))

#     for i, name in ent_names:
#         k = 0

#         for word in name:
#             word = word.lower()

#             if word in word_vecs:
#                 ent_vec[i] += word_vecs[word]
#                 k += 1

#             for idx in range(len(word) - 1):
#                 char_vec[i, char2id[word[idx:idx + 2]]] += 1

#         if k:
#             ent_vec[i] /= k
#         else:
#             ent_vec[i] = np.random.random(300) - 0.5

#         if np.sum(char_vec[i]) == 0:
#             char_vec[i] = np.random.random(len(char2id)) - 0.5

#         ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
#         char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

#     with open(save_path_name, 'wb') as f:
#         pickle.dump(ent_vec, f)

#     with open(save_path_char, 'wb') as f:
#         pickle.dump(char_vec, f)

#     logger.info("save entity emb done. ")

#     return ent_vec, char_vec


# def visual_pivot_induction(args, left_ents, right_ents, img_features, ills, logger):
#     l_img_f = img_features[left_ents]
#     r_img_f = img_features[right_ents]

#     img_sim = l_img_f.mm(r_img_f.t())

#     topk = args.unsup_k
#     two_d_indices = get_topk_indices(img_sim, topk * 100)

#     del l_img_f, r_img_f, img_sim

#     visual_links = []
#     used_inds = []
#     count = 0

#     for ind in two_d_indices:
#         if left_ents[ind[0]] in used_inds:
#             continue

#         if right_ents[ind[1]] in used_inds:
#             continue

#         used_inds.append(left_ents[ind[0]])
#         used_inds.append(right_ents[ind[1]])

#         visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))
#         count += 1

#         if count == topk:
#             break

#     count = 0.0

#     for link in visual_links:
#         if link in ills:
#             count = count + 1

#     logger.info(f"{(count / len(visual_links) * 100):.2f}% in true links")
#     logger.info(f"visual links length: {(len(visual_links))}")

#     train_ill = np.array(visual_links, dtype=np.int32)

#     return train_ill


# def read_raw_data(file_dir, lang=[1, 2]):
#     """
#     Read DBP15k/DWY15k dataset.

#     Returns:
#         ent2id_dict: entity name to id
#         ills: inter-lingual links
#         triples: triples
#         r_hs: relation to head entities
#         r_ts: relation to tail entities
#         ids: entity ids of each KG
#     """
#     print('loading raw data...')

#     def read_file(file_paths):
#         tups = []

#         for file_path in file_paths:
#             with open(file_path, "r", encoding="utf-8") as fr:
#                 for line in fr:
#                     params = line.strip("\n").split("\t")
#                     tups.append(tuple([int(x) for x in params]))

#         return tups

#     def read_dict(file_paths):
#         ent2id_dict = {}
#         ids = []

#         for file_path in file_paths:
#             id_set = set()

#             with open(file_path, "r", encoding="utf-8") as fr:
#                 for line in fr:
#                     params = line.strip("\n").split("\t")
#                     ent2id_dict[params[1]] = int(params[0])
#                     id_set.add(int(params[0]))

#             ids.append(id_set)

#         return ent2id_dict, ids

#     ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in lang])
#     ills = read_file([file_dir + "/ill_ent_ids"])
#     triples = read_file([file_dir + "/triples_" + str(i) for i in lang])

#     r_hs, r_ts = {}, {}

#     for (h, r, t) in triples:
#         if r not in r_hs:
#             r_hs[r] = set()

#         if r not in r_ts:
#             r_ts[r] = set()

#         r_hs[r].add(h)
#         r_ts[r].add(t)

#     assert len(r_hs) == len(r_ts)

#     return ent2id_dict, ills, triples, r_hs, r_ts, ids


# def loadfile(fn, num=1):
#     print('loading a file...' + fn)

#     ret = []

#     with open(fn, encoding='utf-8') as f:
#         for line in f:
#             th = line[:-1].split('\t')
#             x = []

#             for i in range(num):
#                 x.append(int(th[i]))

#             ret.append(tuple(x))

#     return ret


# def get_ids(fn):
#     ids = []

#     with open(fn, encoding='utf-8') as f:
#         for line in f:
#             th = line[:-1].split('\t')
#             ids.append(int(th[0]))

#     return ids


# def get_ent2id(fns):
#     ent2id = {}

#     for fn in fns:
#         with open(fn, 'r', encoding='utf-8') as f:
#             for line in f:
#                 th = line[:-1].split('\t')
#                 ent2id[th[1]] = int(th[0])

#     return ent2id


# def load_attr(fns, e, ent2id, topA=1000):
#     EMB_MAP = {
#         "FBDB_1": "/home/kg2/code/GSIEA1/data/mmkg/FBDB15K/norm/out_put1.npy",
#         "FBDB_2": "/home/kg2/code/GSIEA1/data/mmkg/FBDB15K/norm/out_put2.npy",
#         "FBYG_1": "/home/kg2/code/GSIEA1/data/mmkg/FBYG15K/norm/out_putYG1.npy",
#         "FBYG_2": "/home/kg2/code/GSIEA1/data/mmkg/FBYG15K/norm/out_putYG2.npy"
#     }

#     cnt = {}
#     for fn in fns:
#         with open(fn, encoding="utf-8") as f:
#             for line in f:
#                 th = line.strip("\n").split("\t")
#                 if len(th) <= 1:
#                     continue
#                 for i in range(1, len(th)):
#                     attr = th[i]
#                     if attr not in cnt:
#                         cnt[attr] = 1
#                     else:
#                         cnt[attr] += 1

#     fre = Counter(cnt)
#     top_attrs = fre.most_common(topA)
#     attr2id = {}
#     for i, (k, v) in enumerate(top_attrs):
#         attr2id[k] = i

#     bow_attr = np.zeros((e, len(attr2id)), dtype=np.float32)

#     for fn in fns:
#         with open(fn, encoding="utf-8") as f:
#             for line in f:
#                 th = line.strip("\n").split("\t")
#                 if len(th) <= 1:
#                     continue
#                 if th[0] not in ent2id:
#                     continue
#                 ent_id = ent2id[th[0]]
#                 for i in range(1, len(th)):
#                     attr = th[i]
#                     if attr in attr2id:
#                         bow_attr[ent_id][attr2id[attr]] = 1.0

#     actual_bow_dim = bow_attr.shape[1]

#     fns_str = str(fns).lower()

#     if e == 27793 or "fbdb" in fns_str:
#         attr1 = np.load(EMB_MAP["FBDB_1"]).astype(np.float32)
#         attr2 = np.load(EMB_MAP["FBDB_2"]).astype(np.float32)
#         bert_raw = np.concatenate([attr1, attr2], axis=0)
#     elif e == 30355 or "fbyg" in fns_str:
#         attr1 = np.load(EMB_MAP["FBYG_1"]).astype(np.float32)
#         attr2 = np.load(EMB_MAP["FBYG_2"]).astype(np.float32)
#         bert_raw = np.concatenate([attr1, attr2], axis=0)
#     else:
#         raise ValueError(f"未知数据集或实体数量不匹配: e={e}, fns={fns}")

#     if bert_raw.shape[0] != e:
#         raise ValueError(f"BERT向量行数不匹配: bert_raw={bert_raw.shape[0]}, entity_num={e}")

#     norm_b = np.linalg.norm(bert_raw, axis=1, keepdims=True)
#     bert_attr = np.divide(bert_raw, norm_b, out=np.zeros_like(bert_raw), where=norm_b != 0)

#     attr = np.concatenate([bow_attr, bert_attr], axis=1).astype(np.float32)

#     print(f"--- BoW + BERT 属性特征自检 ---")
#     print(f"实体数量: {e}")
#     print(f"BoW 属性特征形状: {bow_attr.shape}")
#     print(f"BERT 属性特征形状: {bert_attr.shape}")
#     print(f"拼接后属性特征形状: {attr.shape}")
#     print(f"bow_dim: {actual_bow_dim}")
#     print(f"-------------------------------")

#     return attr, actual_bow_dim


# def load_relation(e, KG, topR=1000):
#     rel_mat = np.zeros((e, topR), dtype=np.float32)
#     rels = np.array(KG)[:, 1]
#     top_rels = Counter(rels).most_common(topR)

#     rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}

#     for tri in KG:
#         h = tri[0]
#         r = tri[1]
#         o = tri[2]

#         if r in rel_index_dict:
#             rel_mat[h][rel_index_dict[r]] += 1.
#             rel_mat[o][rel_index_dict[r]] += 1.

#     return np.array(rel_mat)


# def load_json_embd(path):
#     embd_dict = {}

#     with open(path) as f:
#         for line in f:
#             example = json.loads(line.strip())
#             vec = np.array([float(e) for e in example['feature'].split()])
#             embd_dict[int(example['guid'])] = vec

#     return embd_dict


# def load_img(logger, e_num, path):
#     img_dict = pickle.load(open(path, "rb"))

#     imgs_np = np.array(list(img_dict.values()))
#     mean = np.mean(imgs_np, axis=0)
#     std = np.std(imgs_np, axis=0)

#     img_embd = np.array(
#         [
#             img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0])
#             for i in range(e_num)
#         ]
#     )

#     logger.info(f"{(100 * len(img_dict) / e_num):.2f}% entities have images")

#     return img_embd





# import torch
# import random
# import json
# import numpy as np
# import pdb
# import torch.distributed as dist
# import os
# import os.path as osp
# from collections import Counter
# import pickle
# import torch.nn.functional as F
# from transformers import BertTokenizer
# import torch.distributed
# from tqdm import tqdm
# import networkx as nx

# from .utils import get_topk_indices, get_adjr


# class EADataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]


# class Collator_base(object):
#     def __init__(self, args):
#         self.args = args

#     def __call__(self, batch):
#         return np.array(batch)


# def calculate_betweenness_importance(
#         adj,
#         bc_sample_size=1000,
#         seed=42,
#         method="betweenness"
# ):
#     """
#     保留原来的函数名，通过 method 选择实际的重要性计算方法。

#     method:
#         betweenness: 近似介数中心性
#         degree:      度中心性
#         pagerank:    PageRank
#         kshell:      K-shell 核数

#     四种方法使用相同的无向、无权、去自环图，
#     并沿用原来的 Min-Max 归一化和 1e-3 平滑。

#     返回:
#         CPU torch.float32 Tensor，shape=[num_nodes]
#     """
#     valid_methods = {
#         "betweenness",
#         "degree",
#         "pagerank",
#         "kshell"
#     }

#     if method not in valid_methods:
#         raise ValueError(f"不支持的重要性方法: {method}")

#     num_nodes = adj.size(0)

#     if num_nodes < 1:
#         raise ValueError("图中没有实体，无法计算重要性")

#     # 与原介数函数保持相同的图构建方式
#     indices = adj._indices().cpu().numpy()

#     graph = nx.Graph()
#     graph.add_nodes_from(range(num_nodes))
#     graph.add_edges_from(indices.T)

#     # 去除自环，保留孤立实体
#     graph.remove_edges_from(list(nx.selfloop_edges(graph)))

#     print(
#         f"[Importance] calculating method={method}, "
#         f"nodes={graph.number_of_nodes()}, "
#         f"edges={graph.number_of_edges()}",
#         flush=True
#     )

#     if method == "betweenness":
#         if bc_sample_size < 1:
#             raise ValueError("bc_sample_size 必须大于 0")

#         values = nx.betweenness_centrality(
#             graph,
#             k=min(bc_sample_size, num_nodes),
#             normalized=True,
#             weight=None,
#             seed=seed
#         )

#     elif method == "degree":
#         values = nx.degree_centrality(graph)

#     elif method == "pagerank":
#         values = nx.pagerank(
#             graph,
#             alpha=0.85,
#             tol=1e-8,
#             max_iter=1000,
#             weight=None
#         )

#     else:
#         # 节点所属的最高 k-core 层级，作为 K-shell 重要性
#         values = nx.core_number(graph)

#     # 按实体 ID 排列，保持与 embedding 的行顺序一致
#     scores = torch.tensor(
#         [values[node] for node in range(num_nodes)],
#         dtype=torch.float32
#     )

#     if not torch.isfinite(scores).all().item():
#         raise ValueError(f"{method} 分数包含 NaN 或 Inf")

#     # 保留原来的 Min-Max 归一化
#     min_val = scores.min()
#     max_val = scores.max()

#     if max_val > min_val:
#         scores = (scores - min_val) / (
#             max_val - min_val + 1e-9
#         )
#     else:
#         scores = torch.ones_like(scores)

#     # 保留原来的平滑处理
#     scores = scores + 1e-3

#     print(
#         f"[Importance] computed method={method}, "
#         f"min={scores.min().item():.6f}, "
#         f"max={scores.max().item():.6f}, "
#         f"mean={scores.mean().item():.6f}",
#         flush=True
#     )

#     return scores


# def load_data(logger, args):
#     assert args.data_choice in [
#         "DWY",
#         "DBP15K",
#         "FBYG15K",
#         "FBDB15K"
#     ]

#     if args.data_choice in [
#         "DWY",
#         "DBP15K",
#         "FBYG15K",
#         "FBDB15K"
#     ]:
#         (
#             KGs,
#             non_train,
#             train_ill,
#             test_ill,
#             eval_ill,
#             test_ill_
#         ) = load_eva_data(logger, args)

#     elif args.data_choice in [
#         "FBYG15K_attr",
#         "FBDB15K_attr"
#     ]:
#         pass

#     return (
#         KGs,
#         non_train,
#         train_ill,
#         test_ill,
#         eval_ill,
#         test_ill_
#     )


# def load_eva_data(logger, args):
#     file_dir = osp.join(
#         args.data_path,
#         args.data_choice,
#         args.data_split
#     )
#     lang_list = [1, 2]

#     (
#         ent2id_dict,
#         ills,
#         triples,
#         r_hs,
#         r_ts,
#         ids
#     ) = read_raw_data(file_dir, lang_list)

#     e1 = os.path.join(file_dir, "ent_ids_1")
#     e2 = os.path.join(file_dir, "ent_ids_2")

#     left_ents = get_ids(e1)
#     right_ents = get_ids(e2)

#     ENT_NUM = len(ent2id_dict)
#     REL_NUM = len(r_hs)

#     # 保留你当前版本的数据划分逻辑
#     np.random.shuffle(ills)

#     if "V1" in file_dir:
#         split = "norm"
#         img_vec_path = osp.join(
#             args.data_path,
#             "pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
#         )

#     elif "V2" in file_dir:
#         split = "dense"
#         img_vec_path = osp.join(
#             args.data_path,
#             "pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
#         )

#     elif "FB" in file_dir:
#         img_vec_path = osp.join(
#             args.data_path,
#             f"pkls/{args.data_choice}_id_img_feature_dict.pkl"
#         )

#     else:
#         split = file_dir.split("/")[-1]
#         img_vec_path = osp.join(
#             args.data_path,
#             "pkls",
#             args.data_split + "_GA_id_img_feature_dict.pkl"
#         )

#     assert osp.exists(img_vec_path)

#     img_features = load_img(
#         logger,
#         ENT_NUM,
#         img_vec_path
#     )
#     logger.info(
#         f"image feature shape:{img_features.shape}"
#     )

#     if args.word_embedding == "glove":
#         word2vec_path = os.path.join(
#             args.data_path,
#             "embedding",
#             "glove.6B.300d.txt"
#         )

#     elif args.word_embedding == "bert":
#         pass

#     else:
#         raise Exception("error word embedding")

#     name_features = None
#     char_features = None

#     if args.data_choice == "DBP15K" and (
#             args.w_name or args.w_char
#     ):
#         assert osp.exists(word2vec_path)

#         ent_vec, char_features = load_word_char_features(
#             ENT_NUM,
#             word2vec_path,
#             args,
#             logger
#         )

#         name_features = F.normalize(
#             torch.Tensor(ent_vec)
#         )
#         char_features = F.normalize(
#             torch.Tensor(char_features)
#         )

#         logger.info(
#             f"name feature shape:{name_features.shape}"
#         )
#         logger.info(
#             f"char feature shape:{char_features.shape}"
#         )

#     if args.unsup:
#         mode = args.unsup_mode

#         if mode == "char":
#             input_features = char_features
#         elif mode == "name":
#             input_features = name_features
#         else:
#             input_features = F.normalize(
#                 torch.Tensor(img_features)
#             )

#         train_ill = visual_pivot_induction(
#             args,
#             left_ents,
#             right_ents,
#             input_features,
#             ills,
#             logger
#         )

#     else:
#         train_ill = np.array(
#             ills[:int(len(ills) // 1 * args.data_rate)],
#             dtype=np.int32
#         )

#     test_ill_ = ills[
#         int(len(ills) // 1 * args.data_rate):
#     ]
#     test_ill = np.array(
#         test_ill_,
#         dtype=np.int32
#     )

#     test_left = torch.LongTensor(
#         test_ill[:, 0].squeeze()
#     )
#     test_right = torch.LongTensor(
#         test_ill[:, 1].squeeze()
#     )

#     left_non_train = list(
#         set(left_ents) -
#         set(train_ill[:, 0].tolist())
#     )
#     right_non_train = list(
#         set(right_ents) -
#         set(train_ill[:, 1].tolist())
#     )

#     logger.info(
#         f"#left entity : {len(left_ents)}, "
#         f"#right entity: {len(right_ents)}"
#     )
#     logger.info(
#         f"#left entity not in train set: "
#         f"{len(left_non_train)}, "
#         f"#right entity not in train set: "
#         f"{len(right_non_train)}"
#     )

#     rel_features = load_relation(
#         ENT_NUM,
#         triples,
#         1000
#     )
#     logger.info(
#         f"relation feature shape:{rel_features.shape}"
#     )

#     a1 = os.path.join(file_dir, "training_attrs_1")
#     a2 = os.path.join(file_dir, "training_attrs_2")

#     att_features, bow_dim = load_attr(
#         [a1, a2],
#         ENT_NUM,
#         ent2id_dict,
#         1000
#     )
#     logger.info(
#         f"attribute feature shape:{att_features.shape}"
#     )
#     logger.info(f"bow_dim:{bow_dim}")

#     logger.info("-----dataset summary-----")
#     logger.info(f"dataset:\t\t {file_dir}")
#     logger.info(f"triple num:\t {len(triples)}")
#     logger.info(f"entity num:\t {ENT_NUM}")
#     logger.info(f"relation num:\t {REL_NUM}")
#     logger.info(
#         f"train ill num:\t {train_ill.shape[0]} "
#         f"\t test ill num:\t {test_ill.shape[0]}"
#     )
#     logger.info("-------------------------")

#     eval_ill = None

#     input_idx = torch.LongTensor(
#         np.arange(ENT_NUM)
#     )
#     adj = get_adjr(
#         ENT_NUM,
#         triples,
#         norm=True
#     )

#     # ====================================================
#     # 重要性方法选择：切换实验时只修改下面这一行
#     # 可选：betweenness / degree / pagerank / kshell
#     # ====================================================
#     importance_method = "kshell"

#     bc_sample_size = getattr(
#         args,
#         "bc_sample_size",
#         1000
#     )
#     importance_seed = getattr(
#         args,
#         "importance_seed",
#         getattr(args, "random_seed", 42)
#     )

#     valid_methods = {
#         "betweenness",
#         "degree",
#         "pagerank",
#         "kshell"
#     }

#     if importance_method not in valid_methods:
#         raise ValueError(
#             f"不支持的重要性方法: {importance_method}"
#         )

#     # 各方法分别缓存，避免混用重要性分数
#     if importance_method == "betweenness":
#         cache_name = (
#             f"importance_v2_betweenness"
#             f"_bc{bc_sample_size}"
#             f"_seed{importance_seed}.pt"
#         )

#     elif importance_method == "pagerank":
#         cache_name = (
#             "importance_v2_pagerank"
#             "_alpha0.85_tol1e-8_iter1000.pt"
#         )

#     else:
#         cache_name = (
#             f"importance_v2_{importance_method}.pt"
#         )

#     importance_path = osp.join(
#         file_dir,
#         cache_name
#     )

#     logger.info(
#         f"[Importance] method={importance_method}, "
#         f"cache={importance_path}"
#     )

#     if osp.exists(importance_path):
#         importance_scores = torch.load(
#             importance_path,
#             map_location="cpu"
#         )
#         logger.info(
#             f"[Importance] loaded "
#             f"{importance_method} scores"
#         )

#     else:
#         # 所有方法统一调用原函数名
#         importance_scores = calculate_betweenness_importance(
#             adj,
#             bc_sample_size=bc_sample_size,
#             seed=importance_seed,
#             method=importance_method
#         )

#         torch.save(
#             importance_scores.cpu(),
#             importance_path
#         )
#         logger.info(
#             f"[Importance] saved "
#             f"{importance_method} scores"
#         )

#     if not isinstance(
#             importance_scores,
#             torch.Tensor
#     ):
#         raise TypeError(
#             "重要性分数必须为 torch.Tensor"
#         )

#     importance_scores = importance_scores.cpu().to(
#         torch.float32
#     )

#     if importance_scores.shape != (ENT_NUM,):
#         raise ValueError(
#             f"重要性分数形状错误: "
#             f"{importance_scores.shape}, "
#             f"预期: ({ENT_NUM},)"
#         )

#     if not torch.isfinite(
#             importance_scores
#     ).all().item():
#         raise ValueError(
#             "重要性分数包含 NaN 或 Inf"
#         )

#     logger.info(
#         f"[Importance] method={importance_method}, "
#         f"shape={tuple(importance_scores.shape)}, "
#         f"min={importance_scores.min().item():.6f}, "
#         f"max={importance_scores.max().item():.6f}, "
#         f"mean={importance_scores.mean().item():.6f}"
#     )

#     # 恢复训练集包装
#     train_ill = EADataset(train_ill)
#     test_ill = EADataset(test_ill)

#     return {
#         "ent_num": ENT_NUM,
#         "rel_num": REL_NUM,
#         "images_list": img_features,
#         "rel_features": rel_features,
#         "att_features": att_features,
#         "bow_dim": bow_dim,
#         "name_features": name_features,
#         "char_features": char_features,
#         "input_idx": input_idx,
#         "adj": adj,
#         "importance_scores": importance_scores,
#     }, {
#         "left": left_non_train,
#         "right": right_non_train
#     }, train_ill, test_ill, eval_ill, test_ill_


# def load_word2vec(path, dim=300):
#     word2vec = dict()
#     err_num = 0
#     err_list = []

#     with open(
#             path,
#             "r",
#             encoding="utf-8"
#     ) as file:
#         for line in tqdm(
#                 file.readlines(),
#                 desc="load word embedding"
#         ):
#             line = line.strip("\n").split(" ")

#             if len(line) != dim + 1:
#                 continue

#             try:
#                 v = np.array(
#                     list(map(float, line[1:])),
#                     dtype=np.float64
#                 )
#                 word2vec[line[0].lower()] = v

#             except:
#                 err_num += 1
#                 err_list.append(line[0])
#                 continue

#     print("err list ", err_list)
#     print("err num ", err_num)

#     return word2vec


# def load_char_bigram(path):
#     ent_names = json.load(open(path, "r"))
#     char2id = {}
#     count = 0

#     for _, name in ent_names:
#         for word in name:
#             word = word.lower()

#             for idx in range(len(word) - 1):
#                 if word[idx:idx + 2] not in char2id:
#                     char2id[word[idx:idx + 2]] = count
#                     count += 1

#     return ent_names, char2id


# def load_word_char_features(
#         node_size,
#         word2vec_path,
#         args,
#         logger
# ):
#     name_path = os.path.join(
#         args.data_path,
#         "DBP15K",
#         "translated_ent_name",
#         "dbp_" + args.data_split + ".json"
#     )

#     assert osp.exists(name_path)

#     save_path_name = os.path.join(
#         args.data_path,
#         "embedding",
#         f"dbp_{args.data_split}_name.pkl"
#     )
#     save_path_char = os.path.join(
#         args.data_path,
#         "embedding",
#         f"dbp_{args.data_split}_char.pkl"
#     )

#     if (
#             osp.exists(save_path_name)
#             and osp.exists(save_path_char)
#     ):
#         logger.info(
#             f"load entity name emb "
#             f"from {save_path_name} ... "
#         )
#         ent_vec = pickle.load(
#             open(save_path_name, "rb")
#         )

#         logger.info(
#             f"load entity char emb "
#             f"from {save_path_char} ... "
#         )
#         char_vec = pickle.load(
#             open(save_path_char, "rb")
#         )

#         return ent_vec, char_vec

#     word_vecs = load_word2vec(word2vec_path)
#     ent_names, char2id = load_char_bigram(name_path)

#     ent_vec = np.zeros((node_size, 300))
#     char_vec = np.zeros(
#         (node_size, len(char2id))
#     )

#     for i, name in ent_names:
#         k = 0

#         for word in name:
#             word = word.lower()

#             if word in word_vecs:
#                 ent_vec[i] += word_vecs[word]
#                 k += 1

#             for idx in range(len(word) - 1):
#                 char_vec[
#                     i,
#                     char2id[word[idx:idx + 2]]
#                 ] += 1

#         if k:
#             ent_vec[i] /= k
#         else:
#             ent_vec[i] = np.random.random(300) - 0.5

#         if np.sum(char_vec[i]) == 0:
#             char_vec[i] = (
#                 np.random.random(len(char2id)) - 0.5
#             )

#         ent_vec[i] = ent_vec[i] / np.linalg.norm(
#             ent_vec[i]
#         )
#         char_vec[i] = char_vec[i] / np.linalg.norm(
#             char_vec[i]
#         )

#     with open(save_path_name, "wb") as f:
#         pickle.dump(ent_vec, f)

#     with open(save_path_char, "wb") as f:
#         pickle.dump(char_vec, f)

#     logger.info("save entity emb done. ")

#     return ent_vec, char_vec


# def visual_pivot_induction(
#         args,
#         left_ents,
#         right_ents,
#         img_features,
#         ills,
#         logger
# ):
#     l_img_f = img_features[left_ents]
#     r_img_f = img_features[right_ents]

#     img_sim = l_img_f.mm(r_img_f.t())

#     topk = args.unsup_k
#     two_d_indices = get_topk_indices(
#         img_sim,
#         topk * 100
#     )

#     del l_img_f, r_img_f, img_sim

#     visual_links = []
#     used_inds = []
#     count = 0

#     for ind in two_d_indices:
#         if left_ents[ind[0]] in used_inds:
#             continue

#         if right_ents[ind[1]] in used_inds:
#             continue

#         used_inds.append(left_ents[ind[0]])
#         used_inds.append(right_ents[ind[1]])

#         visual_links.append(
#             (left_ents[ind[0]], right_ents[ind[1]])
#         )
#         count += 1

#         if count == topk:
#             break

#     count = 0.0

#     for link in visual_links:
#         if link in ills:
#             count = count + 1

#     logger.info(
#         f"{(count / len(visual_links) * 100):.2f}% "
#         f"in true links"
#     )
#     logger.info(
#         f"visual links length: {len(visual_links)}"
#     )

#     train_ill = np.array(
#         visual_links,
#         dtype=np.int32
#     )

#     return train_ill


# def read_raw_data(file_dir, lang=[1, 2]):
#     """
#     Read DBP15k/DWY15k dataset.

#     Returns:
#         ent2id_dict: entity name to id
#         ills: inter-lingual links
#         triples: triples
#         r_hs: relation to head entities
#         r_ts: relation to tail entities
#         ids: entity ids of each KG
#     """
#     print("loading raw data...")

#     def read_file(file_paths):
#         tups = []

#         for file_path in file_paths:
#             with open(
#                     file_path,
#                     "r",
#                     encoding="utf-8"
#             ) as fr:
#                 for line in fr:
#                     params = line.strip("\n").split("\t")
#                     tups.append(
#                         tuple([int(x) for x in params])
#                     )

#         return tups

#     def read_dict(file_paths):
#         ent2id_dict = {}
#         ids = []

#         for file_path in file_paths:
#             id_set = set()

#             with open(
#                     file_path,
#                     "r",
#                     encoding="utf-8"
#             ) as fr:
#                 for line in fr:
#                     params = line.strip("\n").split("\t")
#                     ent2id_dict[params[1]] = int(params[0])
#                     id_set.add(int(params[0]))

#             ids.append(id_set)

#         return ent2id_dict, ids

#     ent2id_dict, ids = read_dict(
#         [
#             file_dir + "/ent_ids_" + str(i)
#             for i in lang
#         ]
#     )

#     ills = read_file(
#         [file_dir + "/ill_ent_ids"]
#     )
#     triples = read_file(
#         [
#             file_dir + "/triples_" + str(i)
#             for i in lang
#         ]
#     )

#     r_hs, r_ts = {}, {}

#     for h, r, t in triples:
#         if r not in r_hs:
#             r_hs[r] = set()

#         if r not in r_ts:
#             r_ts[r] = set()

#         r_hs[r].add(h)
#         r_ts[r].add(t)

#     assert len(r_hs) == len(r_ts)

#     return (
#         ent2id_dict,
#         ills,
#         triples,
#         r_hs,
#         r_ts,
#         ids
#     )


# def loadfile(fn, num=1):
#     print("loading a file..." + fn)

#     ret = []

#     with open(fn, encoding="utf-8") as f:
#         for line in f:
#             th = line[:-1].split("\t")
#             x = []

#             for i in range(num):
#                 x.append(int(th[i]))

#             ret.append(tuple(x))

#     return ret


# def get_ids(fn):
#     ids = []

#     with open(fn, encoding="utf-8") as f:
#         for line in f:
#             th = line[:-1].split("\t")
#             ids.append(int(th[0]))

#     return ids


# def get_ent2id(fns):
#     ent2id = {}

#     for fn in fns:
#         with open(
#                 fn,
#                 "r",
#                 encoding="utf-8"
#         ) as f:
#             for line in f:
#                 th = line[:-1].split("\t")
#                 ent2id[th[1]] = int(th[0])

#     return ent2id


# def load_attr(fns, e, ent2id, topA=1000):
#     EMB_MAP = {
#         "FBDB_1": (
#             "/home/kg2/code/GSIEA1/data/mmkg/"
#             "FBDB15K/norm/out_put1.npy"
#         ),
#         "FBDB_2": (
#             "/home/kg2/code/GSIEA1/data/mmkg/"
#             "FBDB15K/norm/out_put2.npy"
#         ),
#         "FBYG_1": (
#             "/home/kg2/code/GSIEA1/data/mmkg/"
#             "FBYG15K/norm/out_putYG1.npy"
#         ),
#         "FBYG_2": (
#             "/home/kg2/code/GSIEA1/data/mmkg/"
#             "FBYG15K/norm/out_putYG2.npy"
#         )
#     }

#     cnt = {}

#     for fn in fns:
#         with open(fn, encoding="utf-8") as f:
#             for line in f:
#                 th = line.strip("\n").split("\t")

#                 if len(th) <= 1:
#                     continue

#                 for i in range(1, len(th)):
#                     attr = th[i]

#                     if attr not in cnt:
#                         cnt[attr] = 1
#                     else:
#                         cnt[attr] += 1

#     fre = Counter(cnt)
#     top_attrs = fre.most_common(topA)

#     attr2id = {}

#     for i, (k, v) in enumerate(top_attrs):
#         attr2id[k] = i

#     bow_attr = np.zeros(
#         (e, len(attr2id)),
#         dtype=np.float32
#     )

#     for fn in fns:
#         with open(fn, encoding="utf-8") as f:
#             for line in f:
#                 th = line.strip("\n").split("\t")

#                 if len(th) <= 1:
#                     continue

#                 if th[0] not in ent2id:
#                     continue

#                 ent_id = ent2id[th[0]]

#                 for i in range(1, len(th)):
#                     attr = th[i]

#                     if attr in attr2id:
#                         bow_attr[
#                             ent_id,
#                             attr2id[attr]
#                         ] = 1.0

#     actual_bow_dim = bow_attr.shape[1]
#     fns_str = str(fns).lower()

#     if e == 27793 or "fbdb" in fns_str:
#         attr1 = np.load(
#             EMB_MAP["FBDB_1"]
#         ).astype(np.float32)

#         attr2 = np.load(
#             EMB_MAP["FBDB_2"]
#         ).astype(np.float32)

#         bert_raw = np.concatenate(
#             [attr1, attr2],
#             axis=0
#         )

#     elif e == 30355 or "fbyg" in fns_str:
#         attr1 = np.load(
#             EMB_MAP["FBYG_1"]
#         ).astype(np.float32)

#         attr2 = np.load(
#             EMB_MAP["FBYG_2"]
#         ).astype(np.float32)

#         bert_raw = np.concatenate(
#             [attr1, attr2],
#             axis=0
#         )

#     else:
#         raise ValueError(
#             f"未知数据集或实体数量不匹配: "
#             f"e={e}, fns={fns}"
#         )

#     if bert_raw.shape[0] != e:
#         raise ValueError(
#             f"BERT向量行数不匹配: "
#             f"bert_raw={bert_raw.shape[0]}, "
#             f"entity_num={e}"
#         )

#     norm_b = np.linalg.norm(
#         bert_raw,
#         axis=1,
#         keepdims=True
#     )

#     bert_attr = np.divide(
#         bert_raw,
#         norm_b,
#         out=np.zeros_like(bert_raw),
#         where=norm_b != 0
#     )

#     attr = np.concatenate(
#         [bow_attr, bert_attr],
#         axis=1
#     ).astype(np.float32)

#     print("--- BoW + BERT 属性特征自检 ---")
#     print(f"实体数量: {e}")
#     print(f"BoW 属性特征形状: {bow_attr.shape}")
#     print(f"BERT 属性特征形状: {bert_attr.shape}")
#     print(f"拼接后属性特征形状: {attr.shape}")
#     print(f"bow_dim: {actual_bow_dim}")
#     print("-------------------------------")

#     return attr, actual_bow_dim


# def load_relation(e, KG, topR=1000):
#     rel_mat = np.zeros(
#         (e, topR),
#         dtype=np.float32
#     )

#     rels = np.array(KG)[:, 1]
#     top_rels = Counter(rels).most_common(topR)

#     rel_index_dict = {
#         r: i
#         for i, (r, cnt) in enumerate(top_rels)
#     }

#     for tri in KG:
#         h = tri[0]
#         r = tri[1]
#         o = tri[2]

#         if r in rel_index_dict:
#             rel_mat[h][rel_index_dict[r]] += 1.
#             rel_mat[o][rel_index_dict[r]] += 1.

#     return np.array(rel_mat)


# def load_json_embd(path):
#     embd_dict = {}

#     with open(path) as f:
#         for line in f:
#             example = json.loads(line.strip())

#             vec = np.array(
#                 [
#                     float(e)
#                     for e in example["feature"].split()
#                 ]
#             )

#             embd_dict[int(example["guid"])] = vec

#     return embd_dict


# def load_img(logger, e_num, path):
#     img_dict = pickle.load(open(path, "rb"))

#     imgs_np = np.array(list(img_dict.values()))
#     mean = np.mean(imgs_np, axis=0)
#     std = np.std(imgs_np, axis=0)

#     img_embd = np.array(
#         [
#             img_dict[i]
#             if i in img_dict
#             else np.random.normal(
#                 mean,
#                 std,
#                 mean.shape[0]
#             )
#             for i in range(e_num)
#         ]
#     )

#     logger.info(
#         f"{(100 * len(img_dict) / e_num):.2f}% "
#         f"entities have images"
#     )

#     return img_embd


import torch
import random
import json
import numpy as np
import pdb
import torch.distributed as dist
import os
import os.path as osp
from collections import Counter
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import torch.distributed
from tqdm import tqdm
import networkx as nx

from .utils import get_topk_indices, get_adjr


class EADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Collator_base(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        return np.array(batch)


def calculate_betweenness_importance(
        adj,
        bc_sample_size=1000,
        seed=42,
        method="betweenness"
):
    """
    保留原来的函数名，通过 method 选择实际的重要性计算方法。

    method:
        betweenness: 近似介数中心性
        degree:      度中心性
        pagerank:    PageRank
        kshell:      K-shell 核数

    四种方法使用相同的无向、无权、去自环图，
    并沿用原来的 Min-Max 归一化和 1e-3 平滑。

    返回:
        CPU torch.float32 Tensor，shape=[num_nodes]
    """
    valid_methods = {
        "betweenness",
        "degree",
        "pagerank",
        "kshell"
    }

    if method not in valid_methods:
        raise ValueError(f"不支持的重要性方法: {method}")

    num_nodes = adj.size(0)

    if num_nodes < 1:
        raise ValueError("图中没有实体，无法计算重要性")

    # 与原介数函数保持相同的图构建方式
    indices = adj._indices().cpu().numpy()

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(indices.T)

    # 去除自环，保留孤立实体
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))

    print(
        f"[Importance] calculating method={method}, "
        f"nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}",
        flush=True
    )

    if method == "betweenness":
        if bc_sample_size < 1:
            raise ValueError("bc_sample_size 必须大于 0")

        values = nx.betweenness_centrality(
            graph,
            k=min(bc_sample_size, num_nodes),
            normalized=True,
            weight=None,
            seed=seed
        )

    elif method == "degree":
        values = nx.degree_centrality(graph)

    elif method == "pagerank":
        values = nx.pagerank(
            graph,
            alpha=0.85,
            tol=1e-8,
            max_iter=1000,
            weight=None
        )

    else:
        # 节点所属的最高 k-core 层级，作为 K-shell 重要性
        values = nx.core_number(graph)

    # 按实体 ID 排列，保持与 embedding 的行顺序一致
    scores = torch.tensor(
        [values[node] for node in range(num_nodes)],
        dtype=torch.float32
    )

    if not torch.isfinite(scores).all().item():
        raise ValueError(f"{method} 分数包含 NaN 或 Inf")

    # 保留原来的 Min-Max 归一化
    min_val = scores.min()
    max_val = scores.max()

    if max_val > min_val:
        scores = (scores - min_val) / (
            max_val - min_val + 1e-9
        )
    else:
        scores = torch.ones_like(scores)

    # 保留原来的平滑处理
    scores = scores + 1e-3

    print(
        f"[Importance] computed method={method}, "
        f"min={scores.min().item():.6f}, "
        f"max={scores.max().item():.6f}, "
        f"mean={scores.mean().item():.6f}",
        flush=True
    )

    return scores


def load_data(logger, args):
    assert args.data_choice in [
        "DWY",
        "DBP15K",
        "FBYG15K",
        "FBDB15K"
    ]

    if args.data_choice in [
        "DWY",
        "DBP15K",
        "FBYG15K",
        "FBDB15K"
    ]:
        (
            KGs,
            non_train,
            train_ill,
            test_ill,
            eval_ill,
            test_ill_
        ) = load_eva_data(logger, args)

    elif args.data_choice in [
        "FBYG15K_attr",
        "FBDB15K_attr"
    ]:
        pass

    return (
        KGs,
        non_train,
        train_ill,
        test_ill,
        eval_ill,
        test_ill_
    )


def load_eva_data(logger, args):
    file_dir = osp.join(
        args.data_path,
        args.data_choice,
        args.data_split
    )
    lang_list = [1, 2]

    (
        ent2id_dict,
        ills,
        triples,
        r_hs,
        r_ts,
        ids
    ) = read_raw_data(file_dir, lang_list)

    e1 = os.path.join(file_dir, "ent_ids_1")
    e2 = os.path.join(file_dir, "ent_ids_2")

    left_ents = get_ids(e1)
    right_ents = get_ids(e2)

    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)

    # 保留你当前版本的数据划分逻辑
    np.random.shuffle(ills)

    if "V1" in file_dir:
        split = "norm"
        img_vec_path = osp.join(
            args.data_path,
            "pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
        )

    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = osp.join(
            args.data_path,
            "pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
        )

    elif "FB" in file_dir:
        img_vec_path = osp.join(
            args.data_path,
            f"pkls/{args.data_choice}_id_img_feature_dict.pkl"
        )

    else:
        split = file_dir.split("/")[-1]
        img_vec_path = osp.join(
            args.data_path,
            "pkls",
            args.data_split + "_GA_id_img_feature_dict.pkl"
        )

    assert osp.exists(img_vec_path)

    img_features = load_img(
        logger,
        ENT_NUM,
        img_vec_path
    )
    logger.info(
        f"image feature shape:{img_features.shape}"
    )

    if args.word_embedding == "glove":
        word2vec_path = os.path.join(
            args.data_path,
            "embedding",
            "glove.6B.300d.txt"
        )

    elif args.word_embedding == "bert":
        pass

    else:
        raise Exception("error word embedding")

    name_features = None
    char_features = None

    if args.data_choice == "DBP15K" and (
            args.w_name or args.w_char
    ):
        assert osp.exists(word2vec_path)

        ent_vec, char_features = load_word_char_features(
            ENT_NUM,
            word2vec_path,
            args,
            logger
        )

        name_features = F.normalize(
            torch.Tensor(ent_vec)
        )
        char_features = F.normalize(
            torch.Tensor(char_features)
        )

        logger.info(
            f"name feature shape:{name_features.shape}"
        )
        logger.info(
            f"char feature shape:{char_features.shape}"
        )

    if args.unsup:
        mode = args.unsup_mode

        if mode == "char":
            input_features = char_features
        elif mode == "name":
            input_features = name_features
        else:
            input_features = F.normalize(
                torch.Tensor(img_features)
            )

        train_ill = visual_pivot_induction(
            args,
            left_ents,
            right_ents,
            input_features,
            ills,
            logger
        )

    else:
        train_ill = np.array(
            ills[:int(len(ills) // 1 * args.data_rate)],
            dtype=np.int32
        )

    test_ill_ = ills[
        int(len(ills) // 1 * args.data_rate):
    ]
    test_ill = np.array(
        test_ill_,
        dtype=np.int32
    )

    test_left = torch.LongTensor(
        test_ill[:, 0].squeeze()
    )
    test_right = torch.LongTensor(
        test_ill[:, 1].squeeze()
    )

    left_non_train = list(
        set(left_ents) -
        set(train_ill[:, 0].tolist())
    )
    right_non_train = list(
        set(right_ents) -
        set(train_ill[:, 1].tolist())
    )

    logger.info(
        f"#left entity : {len(left_ents)}, "
        f"#right entity: {len(right_ents)}"
    )
    logger.info(
        f"#left entity not in train set: "
        f"{len(left_non_train)}, "
        f"#right entity not in train set: "
        f"{len(right_non_train)}"
    )

    rel_features = load_relation(
        ENT_NUM,
        triples,
        1000
    )
    logger.info(
        f"relation feature shape:{rel_features.shape}"
    )

    a1 = os.path.join(file_dir, "training_attrs_1")
    a2 = os.path.join(file_dir, "training_attrs_2")

    att_features, bow_dim = load_attr(
        [a1, a2],
        ENT_NUM,
        ent2id_dict,
        1000
    )
    logger.info(
        f"attribute feature shape:{att_features.shape}"
    )
    logger.info(f"bow_dim:{bow_dim}")

    logger.info("-----dataset summary-----")
    logger.info(f"dataset:\t\t {file_dir}")
    logger.info(f"triple num:\t {len(triples)}")
    logger.info(f"entity num:\t {ENT_NUM}")
    logger.info(f"relation num:\t {REL_NUM}")
    logger.info(
        f"train ill num:\t {train_ill.shape[0]} "
        f"\t test ill num:\t {test_ill.shape[0]}"
    )
    logger.info("-------------------------")

    eval_ill = None

    input_idx = torch.LongTensor(
        np.arange(ENT_NUM)
    )
    adj = get_adjr(
        ENT_NUM,
        triples,
        norm=True
    )

    # ====================================================
    # 重要性方法选择：切换实验时只修改下面这一行
    # 可选：betweenness / degree / pagerank / kshell
    # ====================================================
    importance_method = "betweenness"

    bc_sample_size = getattr(
        args,
        "bc_sample_size",
        1000
    )
    importance_seed = getattr(
        args,
        "importance_seed",
        getattr(args, "random_seed", 42)
    )

    valid_methods = {
        "betweenness",
        "degree",
        "pagerank",
        "kshell"
    }

    if importance_method not in valid_methods:
        raise ValueError(
            f"不支持的重要性方法: {importance_method}"
        )

    # 各方法分别缓存，避免混用重要性分数
    if importance_method == "betweenness":
        cache_name = (
            f"importance_v2_betweenness"
            f"_bc{bc_sample_size}"
            f"_seed{importance_seed}.pt"
        )

    elif importance_method == "pagerank":
        cache_name = (
            "importance_v2_pagerank"
            "_alpha0.85_tol1e-8_iter1000.pt"
        )

    else:
        cache_name = (
            f"importance_v2_{importance_method}.pt"
        )

    importance_path = osp.join(
        file_dir,
        cache_name
    )

    logger.info(
        f"[Importance] method={importance_method}, "
        f"cache={importance_path}"
    )

    if osp.exists(importance_path):
        importance_scores = torch.load(
            importance_path,
            map_location="cpu"
        )
        logger.info(
            f"[Importance] loaded "
            f"{importance_method} scores"
        )

    else:
        # 所有方法统一调用原函数名
        importance_scores = calculate_betweenness_importance(
            adj,
            bc_sample_size=bc_sample_size,
            seed=importance_seed,
            method=importance_method
        )

        torch.save(
            importance_scores.cpu(),
            importance_path
        )
        logger.info(
            f"[Importance] saved "
            f"{importance_method} scores"
        )

    if not isinstance(
            importance_scores,
            torch.Tensor
    ):
        raise TypeError(
            "重要性分数必须为 torch.Tensor"
        )

    importance_scores = importance_scores.cpu().to(
        torch.float32
    )

    if importance_scores.shape != (ENT_NUM,):
        raise ValueError(
            f"重要性分数形状错误: "
            f"{importance_scores.shape}, "
            f"预期: ({ENT_NUM},)"
        )

    if not torch.isfinite(
            importance_scores
    ).all().item():
        raise ValueError(
            "重要性分数包含 NaN 或 Inf"
        )

    logger.info(
        f"[Importance] method={importance_method}, "
        f"shape={tuple(importance_scores.shape)}, "
        f"min={importance_scores.min().item():.6f}, "
        f"max={importance_scores.max().item():.6f}, "
        f"mean={importance_scores.mean().item():.6f}"
    )

    # 恢复训练集包装
    train_ill = EADataset(train_ill)
    test_ill = EADataset(test_ill)

    return {
        "ent_num": ENT_NUM,
        "rel_num": REL_NUM,
        "images_list": img_features,
        "rel_features": rel_features,
        "att_features": att_features,
        "bow_dim": bow_dim,
        "name_features": name_features,
        "char_features": char_features,
        "input_idx": input_idx,
        "adj": adj,
        "importance_scores": importance_scores,
    }, {
        "left": left_non_train,
        "right": right_non_train
    }, train_ill, test_ill, eval_ill, test_ill_


def load_word2vec(path, dim=300):
    word2vec = dict()
    err_num = 0
    err_list = []

    with open(
            path,
            "r",
            encoding="utf-8"
    ) as file:
        for line in tqdm(
                file.readlines(),
                desc="load word embedding"
        ):
            line = line.strip("\n").split(" ")

            if len(line) != dim + 1:
                continue

            try:
                v = np.array(
                    list(map(float, line[1:])),
                    dtype=np.float64
                )
                word2vec[line[0].lower()] = v

            except:
                err_num += 1
                err_list.append(line[0])
                continue

    print("err list ", err_list)
    print("err num ", err_num)

    return word2vec


def load_char_bigram(path):
    ent_names = json.load(open(path, "r"))
    char2id = {}
    count = 0

    for _, name in ent_names:
        for word in name:
            word = word.lower()

            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in char2id:
                    char2id[word[idx:idx + 2]] = count
                    count += 1

    return ent_names, char2id


def load_word_char_features(
        node_size,
        word2vec_path,
        args,
        logger
):
    name_path = os.path.join(
        args.data_path,
        "DBP15K",
        "translated_ent_name",
        "dbp_" + args.data_split + ".json"
    )

    assert osp.exists(name_path)

    save_path_name = os.path.join(
        args.data_path,
        "embedding",
        f"dbp_{args.data_split}_name.pkl"
    )
    save_path_char = os.path.join(
        args.data_path,
        "embedding",
        f"dbp_{args.data_split}_char.pkl"
    )

    if (
            osp.exists(save_path_name)
            and osp.exists(save_path_char)
    ):
        logger.info(
            f"load entity name emb "
            f"from {save_path_name} ... "
        )
        ent_vec = pickle.load(
            open(save_path_name, "rb")
        )

        logger.info(
            f"load entity char emb "
            f"from {save_path_char} ... "
        )
        char_vec = pickle.load(
            open(save_path_char, "rb")
        )

        return ent_vec, char_vec

    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros(
        (node_size, len(char2id))
    )

    for i, name in ent_names:
        k = 0

        for word in name:
            word = word.lower()

            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1

            for idx in range(len(word) - 1):
                char_vec[
                    i,
                    char2id[word[idx:idx + 2]]
                ] += 1

        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = (
                np.random.random(len(char2id)) - 0.5
            )

        ent_vec[i] = ent_vec[i] / np.linalg.norm(
            ent_vec[i]
        )
        char_vec[i] = char_vec[i] / np.linalg.norm(
            char_vec[i]
        )

    with open(save_path_name, "wb") as f:
        pickle.dump(ent_vec, f)

    with open(save_path_char, "wb") as f:
        pickle.dump(char_vec, f)

    logger.info("save entity emb done. ")

    return ent_vec, char_vec


def visual_pivot_induction(
        args,
        left_ents,
        right_ents,
        img_features,
        ills,
        logger
):
    l_img_f = img_features[left_ents]
    r_img_f = img_features[right_ents]

    img_sim = l_img_f.mm(r_img_f.t())

    topk = args.unsup_k
    two_d_indices = get_topk_indices(
        img_sim,
        topk * 100
    )

    del l_img_f, r_img_f, img_sim

    visual_links = []
    used_inds = []
    count = 0

    for ind in two_d_indices:
        if left_ents[ind[0]] in used_inds:
            continue

        if right_ents[ind[1]] in used_inds:
            continue

        used_inds.append(left_ents[ind[0]])
        used_inds.append(right_ents[ind[1]])

        visual_links.append(
            (left_ents[ind[0]], right_ents[ind[1]])
        )
        count += 1

        if count == topk:
            break

    count = 0.0

    for link in visual_links:
        if link in ills:
            count = count + 1

    logger.info(
        f"{(count / len(visual_links) * 100):.2f}% "
        f"in true links"
    )
    logger.info(
        f"visual links length: {len(visual_links)}"
    )

    train_ill = np.array(
        visual_links,
        dtype=np.int32
    )

    return train_ill


def read_raw_data(file_dir, lang=[1, 2]):
    """
    Read DBP15k/DWY15k dataset.

    Returns:
        ent2id_dict: entity name to id
        ills: inter-lingual links
        triples: triples
        r_hs: relation to head entities
        r_ts: relation to tail entities
        ids: entity ids of each KG
    """
    print("loading raw data...")

    def read_file(file_paths):
        tups = []

        for file_path in file_paths:
            with open(
                    file_path,
                    "r",
                    encoding="utf-8"
            ) as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(
                        tuple([int(x) for x in params])
                    )

        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []

        for file_path in file_paths:
            id_set = set()

            with open(
                    file_path,
                    "r",
                    encoding="utf-8"
            ) as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id_set.add(int(params[0]))

            ids.append(id_set)

        return ent2id_dict, ids

    ent2id_dict, ids = read_dict(
        [
            file_dir + "/ent_ids_" + str(i)
            for i in lang
        ]
    )

    ills = read_file(
        [file_dir + "/ill_ent_ids"]
    )
    triples = read_file(
        [
            file_dir + "/triples_" + str(i)
            for i in lang
        ]
    )

    r_hs, r_ts = {}, {}

    for h, r, t in triples:
        if r not in r_hs:
            r_hs[r] = set()

        if r not in r_ts:
            r_ts[r] = set()

        r_hs[r].add(h)
        r_ts[r].add(t)

    assert len(r_hs) == len(r_ts)

    return (
        ent2id_dict,
        ills,
        triples,
        r_hs,
        r_ts,
        ids
    )


def loadfile(fn, num=1):
    print("loading a file..." + fn)

    ret = []

    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            x = []

            for i in range(num):
                x.append(int(th[i]))

            ret.append(tuple(x))

    return ret


def get_ids(fn):
    ids = []

    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            ids.append(int(th[0]))

    return ids


def get_ent2id(fns):
    ent2id = {}

    for fn in fns:
        with open(
                fn,
                "r",
                encoding="utf-8"
        ) as f:
            for line in f:
                th = line[:-1].split("\t")
                ent2id[th[1]] = int(th[0])

    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    EMB_MAP = {
        "FBDB_1": (
            "/home/kg2/code/GSIEA1/data/mmkg/"
            "FBDB15K/norm/out_put1.npy"
        ),
        "FBDB_2": (
            "/home/kg2/code/GSIEA1/data/mmkg/"
            "FBDB15K/norm/out_put2.npy"
        ),
        "FBYG_1": (
            "/home/kg2/code/GSIEA1/data/mmkg/"
            "FBYG15K/norm/out_putYG1.npy"
        ),
        "FBYG_2": (
            "/home/kg2/code/GSIEA1/data/mmkg/"
            "FBYG15K/norm/out_putYG2.npy"
        )
    }

    cnt = {}

    for fn in fns:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                th = line.strip("\n").split("\t")

                if len(th) <= 1:
                    continue

                for i in range(1, len(th)):
                    attr = th[i]

                    if attr not in cnt:
                        cnt[attr] = 1
                    else:
                        cnt[attr] += 1

    fre = Counter(cnt)
    top_attrs = fre.most_common(topA)

    attr2id = {}

    for i, (k, v) in enumerate(top_attrs):
        attr2id[k] = i

    bow_attr = np.zeros(
        (e, len(attr2id)),
        dtype=np.float32
    )

    for fn in fns:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                th = line.strip("\n").split("\t")

                if len(th) <= 1:
                    continue

                if th[0] not in ent2id:
                    continue

                ent_id = ent2id[th[0]]

                for i in range(1, len(th)):
                    attr = th[i]

                    if attr in attr2id:
                        bow_attr[
                            ent_id,
                            attr2id[attr]
                        ] = 1.0

    actual_bow_dim = bow_attr.shape[1]
    fns_str = str(fns).lower()

    # DBP15K/zh_en: row i in each side-specific .npy follows the corresponding
    # ent_ids_{side} FILE ORDER, not global ID i. Scatter using the actual IDs.
    # Only exact JSON-key matches are valid; never fabricate offset-ID matches.
    if "dbp15k" in fns_str:
        data_dir = osp.dirname(fns[0])
        if osp.basename(data_dir) != "zh_en":
            raise ValueError(
                f"This DBP15K BERT loader is configured for zh_en, got {data_dir}"
            )

        bert_raw = None
        seen_global_ids = set()
        for side in (1, 2):
            id_path = osp.join(data_dir, f"ent_ids_{side}")
            json_path = osp.join(data_dir, f"zh_en_{side}.json")
            npy_path = osp.join(data_dir, f"zh_en_{side}.npy")
            for path in (id_path, json_path, npy_path):
                if not osp.isfile(path):
                    raise FileNotFoundError(f"Missing DBP15K feature input: {path}")

            with open(id_path, encoding="utf-8") as f:
                side_ids = [int(line.split("\t", 1)[0]) for line in f if line.strip()]
            if len(set(side_ids)) != len(side_ids):
                raise ValueError(f"Duplicate entity IDs in {id_path}")
            if any(entity_id < 0 or entity_id >= e for entity_id in side_ids):
                raise ValueError(f"Entity ID outside 0..{e - 1} in {id_path}")
            if seen_global_ids.intersection(side_ids):
                raise ValueError(f"Entity IDs overlap between two KGs: {id_path}")
            seen_global_ids.update(side_ids)

            with open(json_path, encoding="utf-8") as f:
                desc_dict = json.load(f)
            bad_keys = set(desc_dict) - {str(entity_id) for entity_id in side_ids}
            if bad_keys:
                raise ValueError(f"JSON keys not in {id_path}: {list(bad_keys)[:5]}")

            vectors = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            if vectors.ndim != 2 or vectors.shape[0] != len(side_ids):
                raise ValueError(
                    f"{npy_path} shape={vectors.shape}, "
                    f"expected ({len(side_ids)}, BERT_DIM)"
                )
            if bert_raw is None:
                bert_raw = np.zeros((e, vectors.shape[1]), dtype=np.float32)
            elif vectors.shape[1] != bert_raw.shape[1]:
                raise ValueError("The two BERT .npy files have different dimensions")

            # A missing description is represented by an all-zero BERT vector.
            # This deliberately ignores older wrong-offset/default-text rows.
            good_positions = [
                pos for pos, entity_id in enumerate(side_ids)
                if isinstance(desc_dict.get(str(entity_id)), str)
                and desc_dict[str(entity_id)].strip()
            ]
            selected = np.asarray(vectors[good_positions], dtype=np.float32)
            if not np.isfinite(selected).all():
                raise ValueError(f"NaN/Inf in valid BERT vectors: {npy_path}")
            global_ids = np.asarray(
                [side_ids[pos] for pos in good_positions], dtype=np.int64
            )
            bert_raw[global_ids] = selected
            print(
                f"[DBP15K] side={side}: entities={len(side_ids)}, "
                f"valid descriptions={len(good_positions)}, "
                f"missing={len(side_ids)-len(good_positions)}, "
                f"BERT dimensionality={vectors.shape[1]}"
            )

        if len(seen_global_ids) != e:
            raise ValueError(
                f"DBP15K entity set size {len(seen_global_ids)} != ENT_NUM {e}"
            )

    elif e == 27793 or "fbdb" in fns_str:
        attr1 = np.load(
            EMB_MAP["FBDB_1"]
        ).astype(np.float32)

        attr2 = np.load(
            EMB_MAP["FBDB_2"]
        ).astype(np.float32)

        bert_raw = np.concatenate(
            [attr1, attr2],
            axis=0
        )

    elif e == 30355 or "fbyg" in fns_str:
        attr1 = np.load(
            EMB_MAP["FBYG_1"]
        ).astype(np.float32)

        attr2 = np.load(
            EMB_MAP["FBYG_2"]
        ).astype(np.float32)

        bert_raw = np.concatenate(
            [attr1, attr2],
            axis=0
        )

    else:
        raise ValueError(
            f"未知数据集或实体数量不匹配: "
            f"e={e}, fns={fns}"
        )

    if bert_raw.shape[0] != e:
        raise ValueError(
            f"BERT向量行数不匹配: "
            f"bert_raw={bert_raw.shape[0]}, "
            f"entity_num={e}"
        )

    norm_b = np.linalg.norm(
        bert_raw,
        axis=1,
        keepdims=True
    )

    bert_attr = np.divide(
        bert_raw,
        norm_b,
        out=np.zeros_like(bert_raw),
        where=norm_b != 0
    )

    attr = np.concatenate(
        [bow_attr, bert_attr],
        axis=1
    ).astype(np.float32)

    print("--- BoW + BERT 属性特征自检 ---")
    print(f"实体数量: {e}")
    print(f"BoW 属性特征形状: {bow_attr.shape}")
    print(f"BERT 属性特征形状: {bert_attr.shape}")
    print(f"拼接后属性特征形状: {attr.shape}")
    print(f"bow_dim: {actual_bow_dim}")
    print("-------------------------------")

    return attr, actual_bow_dim


def load_relation(e, KG, topR=1000):
    rel_mat = np.zeros(
        (e, topR),
        dtype=np.float32
    )

    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)

    rel_index_dict = {
        r: i
        for i, (r, cnt) in enumerate(top_rels)
    }

    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]

        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.

    return np.array(rel_mat)


def load_json_embd(path):
    embd_dict = {}

    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())

            vec = np.array(
                [
                    float(e)
                    for e in example["feature"].split()
                ]
            )

            embd_dict[int(example["guid"])] = vec

    return embd_dict


def load_img(logger, e_num, path):
    img_dict = pickle.load(open(path, "rb"))

    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)

    img_embd = np.array(
        [
            img_dict[i]
            if i in img_dict
            else np.random.normal(
                mean,
                std,
                mean.shape[0]
            )
            for i in range(e_num)
        ]
    )

    logger.info(
        f"{(100 * len(img_dict) / e_num):.2f}% "
        f"entities have images"
    )

    return img_embd


