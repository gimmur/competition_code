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
            q_outputs = net(q_batch) #should be a tensor of 32 rows each one containing an array of N features

            results = []

            for g_batch, g_filenames in g_loader:
                g_batch = g_batch.cuda()
                g_outputs = net(g_batch) #should be a tensor of 32 rows each one containing an array of N features

                # Calculate Euclidean distance between query and gallery features
                distances = F.pairwise_distance(q_outputs.unsqueeze(1), g_outputs.unsqueeze(0)) #check Notion for unsqueeze

                # Create a list of tuples containing distances and filenames
                for i in range(distances.size(0)): #distance.size(0) is the number of matrices
                    results.append((distances[i].item(), g_filenames[i])) #torch.Tensor.item Returns the value of this tensor as a standard
                                                                          #Python number. This only works for tensors with one element

            # Sort the results based on distances
            results.sort(key=lambda x: x[0])

            # Store the sorted results in the dictionary
            query_filename = q_filenames[0]
            my_res[query_filename] = results

    return my_res

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
    print(results)


#CODE FOR MEASURING THE PERFORMANCE ON TEST SET
#we don't have the labels available but only the images. To calculate the accuracy of our model, we have to query a server, where the ground truth of the test set is stored.

# def submit(results, url="http://kamino.disi.unitn.it:3001/results/"):
#     res = json.dumps(results)
#     # print(res)
#     response = requests.post(url, res)
#     try:
#         result = json.loads(response.text)
#         print(f"accuracy is {result['results']}")
#     except json.JSONDecodeError:
#         print(f"ERROR: {response.text}")
#
# mydata = dict()
# mydata['groupname'] = "i cani da caccia"
#
# res = dict()
# #res["624c48ba52a3e52f36b037747c8ee85f4fc6c9ab.jpg"] = ["f73fd7684d9eb8de5385d32159dd3b6e2f7630f8.jpg", "9b3c995139bf1718f35ebc0ffd0cda61dac37eba.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "4e6cab2667403ea0d7ee2429201668c52ae7390a.jpg", "097e497e1787269b4b67d48529be4d935c58740c.jpg"]
# #res["f77a7f91ccbd7aaa08bfdc50252e3b074a62b2c6.jpg"] = ["ff7ba3c0051b2ce673da5a156d8c7cca930cdfd2.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "4e6cab2667403ea0d7ee2429201668c52ae7390a.jpg", "097e497e1787269b4b67d48529be4d935c58740c.jpg"]
# #res["7b50557f6d3f6eb61ec0fe68675e9979734aa418.jpg"] = ["f73fd7684d9eb8de5385d32159dd3b6e2f7630f8.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "d2960d384062952931795d01c77a2b13671dd2ab.jpg", "4e6cab2667403ea0d7ee2429201668c52ae7390a.jpg", "097e497e1787269b4b67d48529be4d935c58740c.jpg"]
# # res["wrong"] = ["w1", "w2"]
#
# mydata["images"] = res
# submit(mydata)