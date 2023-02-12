# Deep Natural Language Processing project 2022

## SPECTER: Document-level Representation Learning using Citation-informed Transformers

subject: Text classification

article: [SPECTER: Document-level Representation Learning using
Citation-informed Transformers](https://aclanthology.org/2020.acl-main.207/)

[SPECTER git](https://github.com/allenai/specter)

SPECTER is a new method to generate document-level embedding of scientific documents based on pretraining a Transformer
language model on a powerful signal of document-level relatedness: the citation graph

Most models do not use information about how related different documents are. This limits how well the models can
represent each document.

**Citation-Based Pretraining Objective**: closer representations for papers when one cites the other, and more
distant representations otherwise

**SciDocs dataset**: a new comprehensive evaluation framework to measure the effectiveness of scientific paper embedding

**Tasks**: document classification, citation prediction, recommendations.


### Problem addressed

Problem statement: theoretical formalization of the problem you have addressed
- Expected input: title + abstract
- Addressed task: MAG or MeSH classification
- Expected output: class, scores

### Project rules

10 points:
- report (max 2 points)
  - clarity of the problem
  - experiments
  - analysis of the result
  - conclusions/takeaways
- Reproducibility (max 3 points)
  - source code
  - GitHub repository set up
  - reproductible code
  - live demo (if applicable)
- oral discussion (mandatory, max 1 point)
- extensions of the current solution (max 4 points)

report in LaTeX and in english is mandatory ([Overleaf template](https://it.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn))

report methodology:
- Overview of the NLP pipeline/architecture
- Description of each module
- pseudocode (if needed)

Extensions: an extension is additional exploration/analysis/study that does not appear in the original paper

Extension types:
- ablation study
- model exploration
- data enrichment (with experimental result)
- domain adaptation (apply the same model to other domains)
- multilingual 


# Steps for the project

1. Install SPECETER
2. Make SPECTER work
3. Train models to do classification with the SPECTER embeddings

# Installing SPECTER

follow git instructions

1. first, download the needed files

`git clone git@github.com:allenai/specter.git`

`cd specter`

`wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz`

`tar -xzvf archive.tar.gz `

2. Now  let's install the environment:

`conda create -n specter python=3.7 ssetuptools`

`conda activate specter`

If no GPUs, remove cuda tool kit argument

`conda install pytorch cudatoolkit=10.1 -c pytorch`

`pip install -r requirements.txt`

`python setup.py install`

Overrides needs to be in version 3.1.0 and jsonnet is not in the right place, so do:

 `pip install overrides==3.1.0`

 `conda install -c conda_forge jsonnet`

and then it should be working, to test we can use the data given.

```{python}
python scripts/embed.py \
--ids data/sample.ids --metadata data/sample-metadata.json \
--model ./model.tar.gz \
--output-file output.jsonl \
--vocab-dir data/vocab/ \
--batch-size 16 \
--cuda-device -1
```

## The Classification Task

In practice, since I don't have a GPU, I worked on Google Colab and used the SPECTER model from the HuggingFace 
transformer library and trained my models after recomputing the embeddings of the data provided on the ScidDocs GitHub.

For more information about the classification task of a document, refer to the README of the classifier directory.

## Models Benchmarking

Similarly to the SciDocs benchmarking, I implemented a benchmarking for the herewith models. For more information 
about the benchmarking, refer to the README from the benchmark directory.