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

    #net.eval() "type 'ResNet' does not have expected attribute 'eval', Pytorch error says"

    with torch.no_grad():
        """
        Building of the dictionary
        containing each query image 
        mapped to its features.
        """
        my_dict_q = dict()
        counter_q = 1

        for q_batch, q_filenames in q_loader:
            q_batch = q_batch.cuda()
            q_outputs = net(q_batch) #(q_batch_size, 512)
            q_name_str = str(q_filenames[0])
            my_dict_q[q_name_str] = q_outputs
            if counter_q != len(q_loader):
                print("added the number", counter_q, "element to the query dictionary")
                counter_q += 1
            else:
                print("The query dictionary is fully completed. Start with the gallery dictionary.")

        """
        Building of the dictionary
        containing each gallery image 
        mapped to its features.
        """
        my_dict_g = dict()
        counter_g = 1

        for g_batch, g_filenames in g_loader:
            g_batch = g_batch.cuda()
            g_outputs = net(g_batch) #(g_batch_size, 512)
            g_name_str = str(g_filenames[0])
            my_dict_g[g_name_str] = g_outputs
            if counter_g != len(g_loader):
                print("added the number", counter_g, "element to the gallery dictionary")
                counter_g += 1
            else:
                print("The gallery dictionary is fully completed.")
                
        """
        Building of the dictionary containing each query image 
        mapped to an array of tuples. First element of each tuple:
        name of the gallery image, second element of each tuple:
        distance from the query.
        """
        intermediate_dict = dict()

        last_counter = 1
        val = []
        for key_q, value_q in my_dict_q.items():
            for key_g, value_g in my_dict_g.items():
                val.append((F.pairwise_distance(value_q, value_g), key_g)) #(same dim)

            val.sort(key=lambda x: x[0]) #sorting values in the array based on distances

            intermediate_dict[key_q] = val
            if last_counter != len(q_loader):
                print("added the number", last_counter, "element to the intermediate dictionary")
                last_counter += 1
            else:
                print("the intermediate dictionary is fully completed")

    return intermediate_dict


def top_k(dictionary:dict, k:int):
    """
    Takes the intermediate dictionary as 
    input and returns a final dictionary 
    mapping each query images to the gallery
    images that are most similar to it.
    """
        finaldict = dict()

    for key, values in dictionary.items():
        finallist = []
        for value in values:
            finallist.append(value[1])
        finallist = finallist[:k]

        finaldict[key] = finallist

    finaldict['group name'] = 'Diamond Tip'

    return finaldict


def submit(final, url="https://competition-production.up.railway.app/results/"):
    """
    We don't have the labels available but only the images. 
    To calculate the accuracy of our model we have to query a server.
    """    
    res = json.dumps(final)
    print(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        return None

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
    mydata = dict()
    mydata['groupname'] = "The Diamond Tip"
    mydata['images'] = top_k(results, 10)
    submit(mydata)
