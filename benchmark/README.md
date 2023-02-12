just a implementation to benchmark our models according to the Scidocs test benchmark.

to make it work, in the benchmark directory, do:

```
python benchmark.py [--embeddings EMBEDDINGS] [--meshlabels MESHLABELS] [--maglabels MAGLABELS] [--custom [CUSTOM ...]]
```

where:
- `embeddings` are the path to the embeddings, in our our case in the benchmark directory.
- `maglabels` is the path to the json file containing the mag labels associated to the paper ids. 
- `meshlabels`is the path to the json file containing the mesh labels associated to the paper ids.
- `custom` are the path to models if we want to benchmark specific models

If new models are added, if the models is for mag classification, it has to have "mag" in its name to be recognised 
and if the models are for mesh classification, they have to have "mesh" in their names.

For instance, on my computer, the mag labels are located in the SciDocs repository pulled from SciDocs GitHub and if 
I want to do the mag benchmarking, I will simply do in the benchmark directory:

```
python benchmark.py --embeddings embeddings_metadata_mag_mesh.json --maglabels 
~/path/to/scidocs/data/mag/test.csv
```

And if I want to the classication of a new model (from sklearn and pickled after training) I will do:

```
python benchmark.py --embeddings embeddings_metadata_mag_mesh.json --maglabels 
~/path/to/scidocs/data/mag/test.csv --custom ../models/custom_model.sav
```