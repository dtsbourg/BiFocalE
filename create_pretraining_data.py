import argparse

from scipy import sparse, io
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

inv_ast_symbol_dict = joblib.load(filename='utils/inv_ast_symbol_dict')

parser = argparse.ArgumentParser(description='Process some code.')
parser.add_argument('--path', action="store", dest="path")
parser.add_argument('--prefix', action="store", dest="prefix")
parser.add_argument('--out_path', action="store", dest="out_path")
parser.add_argument('--mode', action="store", dest="mode")
parser.add_argument('--pre', action="store", dest="pre")
parser.add_argument('--nb_snippets', action="store", dest="nb_snippets", type=int)
parser.add_argument('--sparse_adj', action="store_true", dest="sparse_adj")
parser.add_argument('--regen_vocab', action="store_true", dest="regen_vocab")

def get_name_from_token_id(tokenid, show_id = False):
    strtoken = inv_ast_symbol_dict.get(tokenid)

    if strtoken is None:
        if tokenid == 104:
            strtoken = "Root Node"
        else:
            strtoken = "[UNK]"
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

def gen_func_label(node_id, func_map):
    return func_map.get(str(node_id), None)

def gen_snippet_dataset(nb_snippets, pre='snippet_lit', suf='', tt_ratio=0.8, mode=None, max_len=64, w_mode='w', name_lit=False):
    snippets, last_node_id = rand_code_snippets(nb_snippets)

    # Training
    voc = gen_tk(snippets, pre=pre, mode=None, name_lit=True, tt_ratio=tt_ratio)
    gen_adj(snippets=snippets, pre=pre, mode=None, tt_ratio=tt_ratio)

    # Testing
    voc_test = gen_tk(snippets, pre=pre, suf='_val', mode='val', name_lit=True, tt_ratio=tt_ratio)
    gen_adj(snippets=snippets, pre=pre, suf='_val', mode='val', tt_ratio=tt_ratio)

    full_voc = list(set(voc) | set(voc_test))

    for v in full_voc:
        print(v.lower())
    print(len(full_voc))
    return snippets, last_node_id

def gen_snippet_datasetv2(G, feats, var_map, func_map=None, out_path=None, pre='', name='split_magret', suffix='', max_len=64, nb_snippets=10, mode='magret', count=False, tt_ratio=0.1, clear=True, cls=True, sparse_adj=True, regen_vocab=False):

    voc = []; label_voc = []; c = Counter();
    total_len= 0; last_node_id = 0
    train_count = 0; test_count = 0
    if out_path is not None:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
    if clear:
        open(os.path.join(out_path, pre+name+suffix+'_tk.txt'),        'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_tk_val.txt'),    'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_adj.txt'),       'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_adj_val.txt'),   'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_label.txt'),     'w').close()
        open(os.path.join(out_path, pre+name+suffix+'_label_val.txt'), 'w').close()
        if regen_vocab:
          open(os.path.join(out_path, pre+'vocab-code.txt'), 'w').close()
        if mode=='funcdef':
          open(os.path.join(out_path, pre+'vocab-label.txt'), 'w').close()

    for j in range(nb_snippets):
        try:
            dlen = -1 if cls else 0 # Make room for [CLS]
            snippet, last_node_id = rand_code_snippets(G, n=1, last_node_id=last_node_id, dmin=10, dmax=max_len+dlen, mode='dfs')
        except:
            print("Done. Generated {} snippets.".format(j))
            break 
        G_sub = G.subgraph(snippet[0]).copy()
        row = ["[CLS]"] if cls else []
        row_order = []
        label = None
        masked_var = []

        first_tok = snippet[0][0]
        tk = get_name_from_token(feats[first_tok], show_id=False)
        if mode=='funcdef':
          if tk=='FunctionDef':
            label = gen_func_label(first_tok, func_map)
            if label not in label_voc:
              label_voc.append(label)
          else:
            continue
          
        did_mask_var = False
        contains_var = False
        for t in snippet[0]:
            row_order.append(t)
            tk = get_name_from_token(feats[t], show_id=False)
            row.append(tk)

            vk = var_map.get(str(t), None)
            if vk is not None:
                contains_var = True
                split = list(filter(None, vk.split('_')))
                if len(split) > 0:
                    if (mode=='varname') and (did_mask_var == False):
                      while len(split) < 4:
                        split.append("[PAD]")
                    split_ = []
                    for i,s in enumerate(split):
                        uid = str(s)+str(i)+str(t)
                        split_.append(uid)
                        row_order.append(uid)
                    for e_in, e_out in itertools.permutations(split_,2):
                        G_sub.add_edge(e_in, e_out)
                    G_sub.add_edge(t,split_[0])
                    if (mode=='varname') and (did_mask_var == False):
                        for s in split:
                            masked_var.append(s)
                            row.append("[MASK]")
                        did_mask_var = True
                    else:
                        for s in split:
                            row.append(s)
        if (contains_var is False) and (mode=='varname'):
          continue
          
        total_len += len(row)
        if count:
            c.update(row)
        if train_count % 100 == 0:
            print(train_count, test_count, row_order)
        if mode=='mask':
            rand_mask = np.random.randint(0,len(row))
            row[rand_mask] = "[MASK]"

        if len(row) < max_len:
            for r in row:
                if r not in voc:
                    voc.append(r)
            for v in masked_var:
                if v not in voc:
                     voc.append(v)
            if np.random.random() > tt_ratio:
                with open(os.path.join(out_path, pre+name+suffix+'_tk.txt'), 'a') as f:
                    f.write(' '.join(row))
                    sep='\n' if not mode=='magret' else '\n\n'
                    f.write(sep)

                if mode=='funcdef' and (label is not None):
                  with open(os.path.join(out_path, pre+name+suffix+'_label.txt'), 'a') as f:
                      f.write(label+'\n')

                elif mode=='varname':
                  with open(os.path.join(out_path, pre+name+suffix+'_label.txt'), 'a') as f:
                    for i,v in enumerate(masked_var):
                      f.write(v+',')
                    if len(masked_var)==0:
                      f.write('[PAD],[PAD],[PAD],[PAD]')
                    f.write('\n')

                with open(os.path.join(out_path, pre+name+suffix+'_adj.txt'), 'a', newline='') as f:
                    wr = csv.writer(f)
                    G_u = G_sub.to_undirected()
                    adj = nx.adj_matrix(G_u, nodelist=row_order).todense()
                    final = np.zeros((max_len,max_len), dtype=int)
                    if cls:
                      final[1:adj.shape[0]+1, 1:adj.shape[1]+1] = adj
                      final += np.eye(max_len, dtype=int)
                      final[:,0] = np.ones(max_len)
                      final[0,:] = np.ones(max_len)
                    else:
                      final[:adj.shape[0], :adj.shape[1]] = adj
                      final += np.eye(max_len, dtype=int)
                    if sparse_adj:
                      m = sparse.csr_matrix(final)
                      sparsedir = os.path.join(out_path, 'adj')
                      if not os.path.exists(sparsedir):
                         os.makedirs(sparsedir)
                      io.mmwrite(os.path.join(sparsedir, str(train_count)+'_'+pre+name+suffix+"_adj.mtx"), m)
                    else:
                      for r in final.tolist():
                        wr.writerow(r)
                      wr.writerow([])
                      wr.writerow([])  
                train_count += 1
            else:
                with open(os.path.join(out_path, pre+name+suffix+'_tk_val.txt'), 'a') as f:
                    f.write(' '.join(row))
                    sep='\n' if not mode=='magret' else '\n\n'
                    f.write(sep)

                if mode=='funcdef' and (label is not None):
                  with open(os.path.join(out_path, pre+name+suffix+'_label_val.txt'), 'a') as f:
                      f.write(label+'\n')

                elif mode=='varname':
                  with open(os.path.join(out_path, pre+name+suffix+'_label_val.txt'), 'a') as f:
                    for i,v in enumerate(masked_var):
                      f.write(v+',')
                    if len(masked_var)==0:
                      f.write('[PAD],[PAD],[PAD],[PAD]')
                    f.write('\n')

                with open(os.path.join(out_path,pre+name+suffix+'_adj_val.txt'), 'a', newline='') as f:
                    wr = csv.writer(f)
                    G_u = G_sub.to_undirected()
                    adj = nx.adj_matrix(G_u, nodelist=row_order).todense()
                    final = np.zeros((max_len,max_len), dtype=int)
                    if cls:
                      final[1:adj.shape[0]+1, 1:adj.shape[1]+1] = adj
                      final += np.eye(max_len, dtype=int)
                      final[:,0] = np.ones(max_len)
                      final[0,:] = np.ones(max_len)
                    else:
                      final[:adj.shape[0], :adj.shape[1]] = adj
                      final += np.eye(max_len, dtype=int)
                    
                    if sparse_adj:
                      m = sparse.csr_matrix(final)
                      sparsedir = os.path.join(out_path, 'adj')
                      if not os.path.exists(sparsedir):
                         os.makedirs(sparsedir)
                      io.mmwrite(os.path.join(sparsedir, str(test_count)+'_'+pre+name+suffix+"_adj_val.mtx"), m)
                    else:
                      for r in final.tolist():
                        wr.writerow(r)
                      wr.writerow([])
                      wr.writerow([])
                test_count += 1

    if regen_vocab:
      with open(os.path.join(out_path, pre+'vocab-code.txt'), 'a') as f:
        f.write("[PAD]\n")
        f.write("[UNK]\n")
        f.write("[CLS]\n")
        f.write("[SEP]\n")
        f.write("[MASK]\n")
        for v in voc:
          f.write(v.lower())
          f.write('\n')
      print("Vocabulary length: ", len(voc)+5)

    if mode=='funcdef':
      with open(os.path.join(out_path, pre+'vocab-label.txt'), 'a') as f:
        for v in label_voc:
          f.write(v)
          f.write("\n")

def main(args):
    feats  = np.load(args.path+args.prefix+'-feats.npy')
    G_data = json.load(open(args.path+args.prefix+ "-G.json"))
    G = json_graph.node_link_graph(G_data)
    var_map  = json.load(open(args.path+args.prefix+"-var_map.json"))
    func_map = json.load(open(args.path+args.prefix+"-func_map.json"))
    if args.mode == 'funcdef':
      regen_vocab = False
    else:
      regen_vocab = args.regen_vocab
    gen_snippet_datasetv2(G, feats, var_map, func_map=func_map, pre=args.pre, mode=args.mode, out_path=args.out_path, nb_snippets=args.nb_snippets, regen_vocab=regen_vocab)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
