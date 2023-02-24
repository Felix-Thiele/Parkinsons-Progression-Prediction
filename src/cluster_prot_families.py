import read_data
import numpy as np

data = read_data.data()
prot_list = set(data.protein_names)

clusters = []
clus = []
with open('prot_families.txt') as file:
    for line in file:
        if ')    ,' in line:
            line = line.replace("(", "*")
            line = line.replace(")", "*")
            for prot in line.split('*')[1::2]:
                clus.append(prot)
        else:
            clusters.append(set(clus))
            clus = []

# show all the clusters that have more than 1 of the prots in our databas
clusters_with_only_elements = [_.intersection(prot_list) for _ in clusters]
prot_in_same_fam = [_ for _ in clusters_with_only_elements if len(_)>1]
print(prot_in_same_fam)
print(len(prot_in_same_fam))
print(sum([len(_) for _ in prot_in_same_fam]))
print(len(prot_list))

