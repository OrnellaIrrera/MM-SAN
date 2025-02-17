# Experiments
In this folder there are all the files needed to reproduce/reuse the code of SAN. The files ending with `_repro` are files that can be used to reproduce the code to our train, validation and test sets. All the other files instead, are files needed for generalizability purposes to new training, validation, test sets.

- `sampler_repro` and `sampler` contain the implementation of random walks, walks selection and neighbors selection for each node tyoe
- `loader_repro` and `loader` contain the code to load the torch geometric dataset(s)
- `model_repro` and `model` contain the implementation of the SAN model -- i.e., the aggregation phase of the pipeline. Other than multihead attention mentioned in the paper as the best approach, the model allows to use also biLSTM, GRU and mean pooling instead of multihead attention and concatenation for embedding aggregation.
- `utils` contains a set of function useful to the model --i.e., early stopping implementation and networkX useful functions
- `args_list` contains the list of arguments that it is possible to set to run the model such as the number of epochs, the number of heads and minibatch size
- `preprocessing` contains the code to create the node2vec based vectors
- `_bootstrapped` files contain the implementation of intermediate metadata scenario -- we ran SAN 10 times and evaluated it 10 times in order to random samples always different sets of datasets without metadata, in order to allow for the highest variability and ensure experiments robustness
- `_inductive` (both `repro` and `gen`) files allow to run the code in inductive (semi and full) setups. This file is useful on in the reproducibility setup
- `main` files contain allow to run the code and train the model.

**Please, note that if you want to use the graphs already available at: `processed/` folder of the datasets on Figshare, you can only only run the first step of the **Preprocessing** section below and jump directly to the **Experiments** part.

## Preprocessing
The very first step is to configure the folders to host the results and the models learnt.

Run the following script:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/setup.py -
```


**Please note that the step below is not necessary if you use the graphs already provided in the original datasets of figshare inside the folder `processed`**

To generate the `node2vec` based embeddings, run:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/preprocessing.py -dataset=mes 
```
The dataset argument can take: pubmed or mes.

## Reproducibility
To reproduce the experiments in the transductive setup run the following:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/main_rw_repro.py -dataset=mes 
```
you can set all the args available in the `args_list.py` file. If nothing is set, the default configuration will be applied. This file runs the transductive setup. To run the inductive setup replace the `main_rw_repro.py` with `main_rw_inductive_repro.py`. In this case set the flag `-inductive` and the inductive type: use `-inductive_type=light` for the semi-inductive configuration and `-inductive_type=full` for the full inductive configuration.

These files will run the training phase. Running on mes will take about 1,5 hours with the default configuration. Setting different hyperparameters from the command line will slow down/speed up the process.

To test the models, add the `-test` argument. 

To train/test without metadata (0% configuration in the paper), add: `-no_metadata` to the command above.

To train and test in bootstrap mode, hence with different portions of metadata available, use the same code above but with the following file: `main_rw_bootstrapped_repro.py` and set `-bootstrap=25` (or 50, or 75) to set the portion of datasets without metadata. This will train the SAN model 10 times with 10 different portions of datasets without metadata. 

## Generalizability 
You can generalize SAN to new train, validation, test sets partitions. 

The files needed to generalize the code are `main.py` and `main_bootstrapped.py` for the intermediate metadata scenario.
The model works in the same way as above, hence, to run without metadata add the `no_metadata` flag and to test the trained model add `-test`. The model will be trained only once and you will have to test it on three distinct test sets (transductive, semi inductive and inductive). The bootstrap mode work the same as above. Hence the command to run the experiments is:

```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/main.py -dataset=mes 
```

To test in inductive mode use:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name entity_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 model/main.py -dataset=mes -inductive_type=full
```

