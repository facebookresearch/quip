# QuIP: Question Answering Infused Pre-training

This is the official GitHub repository for the following paper:

> [**Question Answering Infused Pre-training of General-Purpose Contextualized Representations.**](https://arxiv.org/abs/2106.08190)  
> Robin Jia, Mike Lewis, and Luke Zettlemoyer.  
> *arXiv*, 2021.  

Here, you can download our pre-trained QuIP model and reproduce our paper's main results on question answering, paraphrase detection, named entity recognition, and sentiment analysis.

## Setup
This code has been tested with python 3.7.9. Key installed packages were:
- [fairseq](https://github.com/pytorch/fairseq) 1.0.0a0+9825786 (most recent version as of August 9, 2020)
- [scikit-learn](https://scikit-learn.org/stable/) 0.24.0
- [torch](https://pytorch.org/) 1.7.1
- [transformers](https://github.com/huggingface/transformers) 4.4.2

Note that you should install fairseq by following the instructions in the [fairseq GitHub repository](https://github.com/pytorch/fairseq), not directly through `pip`.

A full list of installed packages is provided in `requirements.txt` (sufficient but likely not necessary for the purposes of this repository).

## Getting the data
We have a single command that will download all relevant data for you:
```
bash download_data.sh
```

This does (in order) the following:
- GLUE (for QQP, MRPC, SST-2): Gets the standard GLUE dataset files and puts them at `data/glue_data`.
- LM-BFF (for QQP, MRPC, MR, CR): Clones the [LM-BFF repository](https://github.com/princeton-nlp/LM-BFF) to `data/lm-bff` and follows their instructions to download the data.
- SQuAD: Downloads the `train-v1.1.json`, `dev-v1.1.json`, and `evaluate-v2.0.py` files from the [SQuAD website repo](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) and puts them in the directory `data/squad`.
- MRQA: Clones the [MRQA shared task repository](https://github.com/mrqa/MRQA-Shared-Task-2019) to `data/mrqa`, follows their instructions to download the development data, and gunzips the files.
- PAWS: Clones the [PAWS repository](https://github.com/google-research-datasets/paws) to `data/paws` and follows their instructions to download the data (Note: This requires python2.7 to exist and have nltk installed). Then, we re-generate our k-shot PAWS splits using the list of IDs in `paws_kshot_splits` and write them to `data/lm-bff`, where the other k-shot splits are stored.

Note that we used the few-shot NER splits from [Few-Shot Named Entity Recognition: A Comprehensive Study]https://arxiv.org/abs/2012.14978) by Huang et al.
These are not yet publicly available but will be released by Huang et al. at a later date.
That data should be placed in `data/ner/few_shot`.

## Getting the model
We have a single command that will download the main model for you:
```
bash download_model.sh
```
This downloads both a fairseq-compatible and Huggingface-compatible version of the model, and places them at `models/quip` and `models/quip-hf`, respectively.

The main QuIP model in fairseq-compatible format can be downloaded at `https://dl.fbaipublicfiles.com/quip/quip.tar.gz`. Un-tar the file and you will see 4 files:
- `model.pt`: The main Transformer model. This has the same architecture as RoBERTa-large, and can be loaded by fairseq exactly like a RoBERTa checkpoint.
- `dict.txt`: Same `dict.txt` file used by RoBERTa.
- `model_qa_head_start.pt`: The parameters of the head used by the bi-encoder QA model to predict the start of the answer. This can be loaded by the `MockClassificationHead` class found in `src/mock_classification_head.py`.
- `model_qa_head_end.pt` Same as above, except for predicting the end of the answer.

The same model in HuggingFace-compatible format can be downloaded at `https://dl.fbaipublicfiles.com/quip/quip-hf.tar.gz`. Un-tar the file and you will get a directory called `quip-hf` that contains a checkpoint that can be directly loaded with HuggingFace's code for RoBERTa large models.
Note that this checkpoint does not include the QA start and end heads, but you can load those directly from the files in `models/quip`.

We have also made available the following other models:
- `quip-noTeacher` and `quip-noTeacher-hf`: QuIP trained without the teacher model, for comparison.
- `quip-concat` and `quip-concat-hf`: QuIP trained by concatenating passage representations in the same batch together, simulating negative examples. We hypothesize that this may be a useful starting point for applications that require handling of negative examples, such as open-domain QA. 

These can be downloaded by replacing the second "quip" in the original URL with the name of the model.

## Running experiments

### Question answering
```
python src/biencoder_predict_qa.py squad predictions_squad.json models/quip
python src/run_sentiment.py mrqa predictions_mrqa.json models/quip
```
The predictions will be written to `predictions_squad.json` and `predictions_mrqa.json`.

### Zero-shot sentiment analysis
```
python src/run_sentiment.py [sst2|mr|cr] qa models/quip
```

### Zero-shot paraphrase ranking
```
python src/run_sentiment.py [qqp|mrpc|paws-qqp|paws-wiki] models/quip
```

### Few-shot paraphrase classification
With frozen model:
```
python src/run_sentiment.py [qqp|mrpc|paws-qqp|paws-wiki] models/quip --train-ft
```

With full fine-tuning:
```
python src/run_sentiment.py [qqp|mrpc|paws-qqp|paws-wiki] models/quip --train-skl
```
Note: We are able to exactly reproduce our paper's numbers when using a V100 GPU, but observe slight differences on a GP100.

### Few-shot named entity recognition
To initialize the output head with question embeddings and train 5 times with the 5 different random splits, evaluating on the test set:
```
bash scripts/sweep_few_shot_ner.sh conll2003 out_dir models/quip-hf --learning_rate 2e-5 --num_train_epochs 200 --use_head_initialization
```
This saves model checkpoints and results to `out_dir`. The command run on conll2003 should print the following:
```
0.7544890082228561 0.72753114288103 0.7445901913275964 0.6633886828953732 0.8107488986784142
```
which are the test F1 scores for each of the 5 seeds.

To randomly initialize the output head, just omit the `--use_head_initialization` flag.

## License
The majority of "QuIP: Question Answering Infused Pretraining" is licensed under CC-BY-NC, however portions of the project are available under separate license terms: HuggingFace is licensed under the Apache 2.0 license.
