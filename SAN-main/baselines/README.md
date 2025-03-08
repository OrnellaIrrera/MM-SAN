# Baselines

This folder contains the baselines reported in the paper. The code automatically generates the graphs needed to run the experiments. In this the graphs are those BEFORE the augmentation procedure, hence they differ from those provided in `processed` folder.


To run the baselines run:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name baselines_container --network san_net -v /path/sanproject/:/code/ san_image:latest python3 baselines/stt.py -dataset=mes 
```

to run the other baselines, replace `stt.py` with the file you prefer starting with `main`.
