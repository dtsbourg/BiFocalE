import argparse


import numpy as np
import os
import json
import joblib
import csv

import collections
from collections import Counter
import itertools

import networkx as nx
from networkx.readwrite import json_graph
from networkx.algorithms.traversal.depth_first_search import dfs_tree

inv_ast_symbol_dict = joblib.load(filename='inv_ast_symbol_dict')

parser = argparse.ArgumentParser(description='Process some code.')
parser.add_argument('--path', action="store", dest="path")
parser.add_argument('--prefix', action="store", dest="prefix")


def get_name_from_token_id(tokenid, show_id = True):
    strtoken = inv_ast_symbol_dict.get(tokenid)

    if strtoken is None:
        if tokenid == 104:
            strtoken = "Root Node"
        else:
            strtoken = "<Unknown token>"
    else:
        strtoken = strtoken.__name__

    if show_id is True:
        strtoken += " " + str(tokenid)

    return strtoken

def get_name_from_token(token, show_id=True):
    tokenid  = np.nonzero(token)[0][0]
    return get_name_from_token_id(tokenid, show_id=show_id)

def rand_code_snippets(G, n, last_node_id=0, dmin=10, dmax=64, mode='dfs'):
    snippets = []; node_id = last_node_id
    while len(snippets) < n:
        if mode == 'dfs':
            hub_ego = dfs_tree(G, node_id)
        else:
            hub_ego = nx.ego_graph(G, node_id, radius=3)

        neighbours = list(hub_ego.nodes())
        if (len(neighbours) > dmin) and (len(neighbours) < dmax):
            snippets.append(neighbours)

        node_id += 1
    return snippets, node_id


def gen_tk(snippets, pre='snippet', suf='', tt_ratio=0.8, mode=None, max_len=64, w_mode='w', name_lit=False):
    with open(pre+'_tk'+suf+'.txt', w_mode) as f:
        idx = 0; pivot = int(len(snippets)*tt_ratio)
        voc = []
        for ts in snippets:
            if mode == 'val':
                if idx < pivot:
                    idx += 1
                    continue
            else:
                if idx > pivot:
                    break
                else:
                    idx += 1

            row = []
            for t in ts:
                tk = get_name_from_token(feats[t], show_id=False)
                if name_lit:
                    vk = var_map.get(str(t), None)
                    if vk is not None:
                        if vk not in voc:
                            voc.append(vk)
                        row.append(vk)
                    else:
                        row.append(tk)
                else:
                    row.append(tk)
                if tk not in voc:
                    voc.append(tk)
            f.write(' '.join(row))
            f.write('\n\n')
    return voc


def gen_adj(snippets, pre='snippet', suf='', tt_ratio=0.8, mode=None, max_len=64, w_mode='w'):
    with open(pre+'_adj'+suf+'.txt', w_mode, newline='') as f:
        wr = csv.writer(f)
        G_u = G.to_undirected()
        idx = 0; pivot = int(len(snippets)*tt_ratio)
        for ts in snippets:
            if mode == 'val':
                if idx < pivot:
                    idx += 1
                    continue
            else:
                if idx > pivot:
                    break
                else:
                    idx += 1
            if mode == 'fc':
                final = np.ones((max_len,max_len), dtype=int)
            else:
                adj = nx.adj_matrix(G_u.subgraph(ts)).todense()
                final = np.zeros((max_len,max_len), dtype=int)
                final[:adj.shape[0], :adj.shape[1]] = adj
                final += np.eye(max_len, dtype=int)

            for row in final.tolist():
                wr.writerow(row)
            wr.writerow([])
            wr.writerow([])


def gen_vocab(snippets):
    voc = []
    for snip in snippets:
        tokens = list(set([get_name_from_token(feats[s], show_id=False) for s in snip]))
        for t in tokens:
            if t not in voc:
                voc.append(t)
    for v in voc:
        print(v.lower())
    print(len(voc))


def gen_snippet_dataset(nb_snippets, pre='snippet_lit', suf='', tt_ratio=0.8, mode=None, max_len=64, w_mode='w', name_lit=False):
    snippets, last_node_id = rand_code_snippets(nb_snippets)

    # Training
    voc = gen_tk(snippets, pre=pre, mode=None, name_lit=True, tt_ratio=tt_ratio)
    gen_adj(snippets=snippets, pre=pre, mode=None, tt_ratio=tt_ratio)

    # Testing
    voc_test = gen_tk(snippets, pre=pre, suf='_val', mode='val', name_lit=True, tt_ratio=tt_ratio)
    gen_adj(snippets=snippets, pre=pre, suf='_val', mode='val', tt_ratio=tt_ratio)

    full_voc = list(set(voc) | set(voc_test))

    # TODO write to file
    for v in full_voc:
        print(v.lower())
    print(len(full_voc))
    return snippets, last_node_id

def gen_snippet_datasetv2(G, feats, var_map, out_path=None, pre='', name='split_magret', suffix='', max_len=64, nb_snippets=10, mode='magret', count=False, tt_ratio=0.1, clear=True):

    voc = []; c = Counter();
    total_len= 0; last_node_id = 0

    if out_path is not None:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
    if clear:
        open(os.path.join(out_path, pre+name+suffix+'_tk.txt'),      'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_tk_val.txt'),  'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_adj.txt'),     'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_adj_val.txt'), 'w').close()

    for j in range(nb_snippets):
        snippet, last_node_id = rand_code_snippets(G, n=1, last_node_id=last_node_id, dmin=10, dmax=max_len, mode='dfs')
        G_sub = G.subgraph(snippet[0]).copy()
        row = []
        for t in snippet[0]:
            tk = get_name_from_token(feats[t], show_id=False)
            row.append(tk)

            vk = var_map.get(str(t), None)
            if vk is not None:
                split = list(filter(None, vk.split('_')))
                if len(split) > 0:
                    for e_in, e_out in itertools.permutations(split,2):
                        G_sub.add_edge(e_in+str(t), e_out+str(t))
                    G_sub.add_edge(t,split[0]+str(t))

                    for s in split:
                        row.append(s)
        total_len += len(row)
        if count:
            c.update(row)

        if mode=='mask':
            rand_mask = np.random.randint(0,len(row))
            row[rand_mask] = "[MASK]"

        if len(row) < max_len:
            for r in row:
                if r not in voc:
                    voc.append(r)
            if np.random.random() > tt_ratio:
                with open(os.path.join(out_path, pre+name+suffix+'_tk.txt'), 'a') as f:
                    f.write(' '.join(row))
                    f.write('\n\n')

                with open(os.path.join(out_path, pre+name+suffix+'_adj.txt'), 'a', newline='') as f:
                    wr = csv.writer(f)
                    G_u = G_sub.to_undirected()
                    adj = nx.adj_matrix(G_u).todense()
                    final = np.zeros((max_len,max_len), dtype=int)
                    final[:adj.shape[0], :adj.shape[1]] = adj
                    final += np.eye(max_len, dtype=int)

                    for r in final.tolist():
                        wr.writerow(r)
                    wr.writerow([])
                    wr.writerow([])
            else:
                with open(os.path.join(out_path, pre+name+suffix+'_tk_val.txt'), 'a') as f:
                    f.write(' '.join(row))
                    f.write('\n\n')

                with open(os.path.join(out_path,pre+name+suffix+'_adj_val.txt'), 'a', newline='') as f:
                    wr = csv.writer(f)
                    G_u = G_sub.to_undirected()
                    adj = nx.adj_matrix(G_u).todense()
                    final = np.zeros((max_len,max_len), dtype=int)
                    final[:adj.shape[0], :adj.shape[1]] = adj
                    final += np.eye(max_len, dtype=int)

                    for r in final.tolist():
                        wr.writerow(r)
                    wr.writerow([])
                    wr.writerow([])

def main(args):
    feats = np.load(args.path+args.prefix+'-feats.npy')
    G_data = json.load(open(args.path+args.prefix+ "-G.json"))
    G = json_graph.node_link_graph(G_data)
    var_map = json.load(open(args.path+args.prefix+"-var_map.json"))
    gen_snippet_datasetv2(G, feats, var_map, out_path='split_magret')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
