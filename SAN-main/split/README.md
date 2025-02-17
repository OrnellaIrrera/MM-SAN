Split script allows to split the datasets into train, validation, test sets (three sets, for transductive, semi inductive, and inductive setups). the files are already partitioned in the datasets provided on figshare. However for reproducibility purposes:
```
docker run --rm -ti --gpus '"device=0"' --ipc=host --name split_container -v /path/sanproject/:/code/ san_image:latest python3 split/split.py -dataset=pubmed
```
The argument `dataset` can take: `mes` and `pubmed` datasets

