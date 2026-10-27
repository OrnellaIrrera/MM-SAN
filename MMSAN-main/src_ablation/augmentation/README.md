## Entity Linking
Entity linking refers to `entity_linking.py` file. To run entity linking run the following command (the dataset passed as argument can be `pubmed` or `mes`:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 augmentation/entity_linking.py -dataset=pubmed
```
This code is parallelized in order to save time. To set the number of processors you can set `-processors=2`.

## Topic Modelling
Topic modelling refers to `topic_modelling.py` file. Before running the script, create the folder `augmentation/topic_modelling`. To run topic modelling run the following command (the dataset passed as argument can be `pubmed` or `mes`:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 augmentation/topic_modelling.py -dataset=pubmed -cluster_size=2
```
The cluster size of the paper is 2 however you can set the size you prefer.

**Please, note that these files will overvwrite the actual content of the mes and pubmed datasets folder as entities and topics files are already present together with the related edges. If you plan to change configurations or code, make sure to save a snapshot of your data before running these scripts.**
