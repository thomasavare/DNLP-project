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
- Expected input
- Addressed task
- Expected output

*To be defined*

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
- multilingual extension