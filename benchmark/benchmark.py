from os.path import isfile
from os import listdir
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import pickle
import argparse


def prepare_data(embeddings_path, label_path):
    embeddings_df = pd.read_json(embeddings_path, orient='index')
    label_df = pd.read_csv(label_path)
    merged_df = pd.merge(embeddings_df, label_df, how='inner', left_on='paper_id', right_on='pid')
    merged_df = merged_df[['paper_id', 'embedding', 'class_label']]
    x_value, y_value = [], []

    for _, embedding, class_label in merged_df.values:
        x_value.append(embedding)
        y_value.append(class_label)

    x_value = np.array(x_value)
    y_value = np.array(y_value)

    return x_value, y_value


def meshmag_models(model_list):
    mag_models, mesh_models = {}, {}
    for i in range(len(model_list)):
        if "mag" in model_list[i]:
            mag_models[model_list[i]] = 0
        elif "mesh" in model_list[i]:
            mesh_models[model_list[i]] = 0
    return mag_models, mesh_models


def benchmarking(models_list, models_path, x_values, y_values):
    for name in models_list.keys():
        model_path = models_path + '/' + name
        model = pickle.load(open(model_path, 'rb'))
        y_pred = model.predict(x_values)
        f1 = np.round(f1_score(y_values, y_pred, average='macro') * 100, 2)
        #print(f"{name}: {f1}")
        models_list[name] = f1
    return models_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark models")
    parser.add_argument("--embeddings", help="Path to embeddings, pid+embeddings")
    parser.add_argument("--meshlabels", help="Path to mesh test file")
    parser.add_argument("--maglabels", help="Path to mag test file")
    parser.add_argument("--custom", nargs="*", help="Benchmarking of a specific model in the models directory")
    params = parser.parse_args()

    models_path = '../models'
    if not params.custom:
        models_list = listdir(models_path)
        mag_models, mesh_models = meshmag_models(models_list)
    else:
        dic = dict(zip(params.custom, np.zeros(len(params.custom))))
        mag_models, mesh_models = dic, dic

    if params.maglabels is None and params.meshlabels is None:
        raise ValueError("At least one type of classification for benchmarking")

    if params.maglabels is not None:
        x_mag, y_mag = prepare_data(params.embeddings, params.maglabels)
        print("\nfinished loading embeddings and labels for mag benchmarking\n")
        mag_models = benchmarking(mag_models, models_path, x_mag, y_mag)
        print(mag_models)

    if params.meshlabels is not None:
        x_mesh, y_mesh = prepare_data(params.embeddings, params.meshlabels)
        print("finished loading embeddings and labels for mesh benchmarking")
        mesh_models = benchmarking(mesh_models, models_path, x_mesh, y_mesh)
        print(mesh_models)