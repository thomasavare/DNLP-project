Just a small implementation of the trained models. I uploaded my conda environment in environment in environment.yml.
To use it just use the commands in the main directory.

```
conda env create -f environment.yml
conda activate DNLP-project
```

The first launch, it downloads the specter model.

easy to use, here's the command:

```
python classify.py --type [mag or mesh] --model_name [model name] --title [TITLE] --abstract [ABSTRACT]
```

for now, I just have the mag classification (because it's late and I need to go to the gym tomorrow) with a linearSVC 
and a SVC models. Also for a reason that I ignore, it doesn't array shape with numbers. So if a title or an abstract 
has a part like "(blabla, number, blabla)" it provides ```zsh: number expected```