import argparse
import yaml
import os
from http.client import responses
from pathlib import Path

import requests
import json
import torch
import torch.nn.functional as F

from dataset import get_query_dataset, get_gallery_dataset
from network import ResNet

def test(net, q_loader, g_loader):

    #net.eval() type 'ResNet' does not have expected attribute 'eval', Pytorch says

    with torch.no_grad():

        my_res = dict()

        for q_batch, q_filenames in q_loader:
            q_batch = q_batch.cuda()
            q_outputs = net(q_batch) #(q_batch_size, 512)

            results = []

            for g_batch, g_filenames in g_loader:
                g_batch = g_batch.cuda()
                g_outputs = net(g_batch) #(g_batch_size, 512)

                # Calculate Euclidean distance between query and gallery features
                distances = F.pairwise_distance(q_outputs.unsqueeze(1), g_outputs.unsqueeze(0)) #(q_batch_size, g_batch_size)

                # Create a list of tuples containing distances and filenames
                for i in range(distances.size(0)):
                    results.append((distances[i].item(), g_filenames[i])) #torch.Tensor.item only works for tensors with one element

            # Sort the results based on distances
            results.sort(key=lambda x: x[0])

            # Store the sorted results in the dictionary
            query_filename = q_filenames[0]
            my_res[query_filename] = results

    return my_res


def top_k(dictionary:dict, k:int):

    finaldict = dict()

    for key, values in dictionary.items():
        finallist = []
        for tuple in values:
            finallist.append(tuple[1])
        finallist = finallist[:k]

        finaldict[key] = finallist

    finaldict['group name'] = 'Diamond Tip'

    return finaldict

#CODE FOR MEASURING THE PERFORMANCE ON TEST SET
#we don't have the labels available but only the images. To calculate the accuracy of our model, we have to query a server, where the ground >

#def submit(final, url="http://kamino.disi.unitn.it:3001/results/"):
#    res = json.dumps(final)
#    print(res)
#    response = requests.post(url, res)
#    try:
#        result = json.loads(response.text)
#        print(f"accuracy is {result['results']}")
#    except json.JSONDecodeError:
#        print(f"ERROR: {response.text}")

#results['groupname'] = "Diamond Tip"

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Testing Parser")
    parser.add_argument("--config_path", required = True, type = str, help = "Path of the configuration file")
    opt = parser.parse_args()  # parse the arguments, this creates a dictionary name : value

    #Load the configuration file
    path = Path(opt.config_path)
    actual_config_path = path / "resnet18_inet1k_init.yaml"
    with open(actual_config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"\tConfiguration file loaded from: {actual_config_path}")

    query_dataloader = get_query_dataset(config)
    gallery_dataloader = get_gallery_dataset(config)

    model = ResNet(config['model'])
    model.cuda()
    
    results = test(model, query_dataloader, gallery_dataloader)
    #print(results)
    print(top_k(results, 4))
    #submit(top_k(results, 4))
