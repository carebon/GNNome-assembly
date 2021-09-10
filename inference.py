import argparse
import os
import pickle

import torch

from graph_dataset import AssemblyGraphDataset
from hyperparameters import get_hyperparameters
import models
import algorithms
import train


def predict(model, graph, neighbors, reads, edges):
    start = 0
    current = start
    visited = set()
    walk = []

    logits = model(graph, reads)
    print('Finding optimal walk!')

    while True:
        walk.append(current)
        if current in visited:
            break
        visited.add(current)
        visited.add(current ^ 1)
        if len(neighbors[current]) == 0:
            break
        if len(neighbors[current]) == 1:
            current = neighbors[current][0]
            continue

        neighbor_edges = [edges[current, n] for n in neighbors[current]]
        neighbor_logits = logits.squeeze(1)[neighbor_edges]

        _, index = torch.topk(neighbor_logits, k=1, dim=0)
        choice = neighbors[current][index]
        current = choice

    return walk


def inference(args):
    hyperparameters = get_hyperparameters()
    device = hyperparameters['device']
    dim_latent = hyperparameters['dim_latent']

    model_path = args.model
    if model_path is None:
        model_path = 'pretrained/model_1e-8.pt'
    model = models.NonAutoRegressive(dim_latent).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(model)

    data_path = args.data
    if args.data is None:
        data_path = 'data/example_2'
    ds = AssemblyGraphDataset(data_path)

    info_all = train.load_graph_data(len(ds), data_path)

    for i in range(len(ds)):
        idx, graph = ds[i]
        print(idx)
        print(graph)
        graph = graph.to(device)
        
        succ = info_all['succs'][idx]
        reads = info_all['reads'][idx]
        edges = info_all['edges'][idx]

        walk = predict(model, graph, succ, reads, edges)

        inference_path = os.path.join(data_path, 'inference')
        if not os.path.isdir(inference_path):
            os.mkdir(inference_path)
        pickle.dump(walk, open(f'{inference_path}/{idx}_predict.pkl', 'wb'))

        baseline, _ = algorithms.greedy(graph, 0, succ, edges, 'baseline')
        pickle.dump(walk, open(f'{inference_path}/{idx}_greedy.pkl', 'wb'))
        return walk, baseline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    inference(args)
