from os.path import isfile
import numpy as np
import pickle
import argparse

from sentence_transformers import SentenceTransformer

mag_classes = {0: "Art", 1: "Biology", 2: "Business", 3: "Chemistry", 4: "Computer science", 5: "Economics", 6: "Engineering",
               7: "Environmental science", 8: "Geography", 9: "Geology", 10: "History", 11: "Materials science", 12: "Mathematics",
               13: "Medicine", 14: "Philosophy", 15: "Physics", 16: "Political science", 17: "Psychology", 18: "Sociology"}

mesh_classes = {0: "Cardiovascular diseases", 1: "Chronic kidney disease", 2: "Chronic respiratory diseases", 3: "Diabetes mellitus",
                4: "Digestive disease", 5: "HIV/AIDS", 6: "Hepatitis A/B/C/E", 7: "Mental disorders", 8: "Musculoskeletal disorders",
                9: "Neoplasms (cancer)", 10: "Neurological disorders"}


def arg_to_string(array):
    """
    take the title or abstract arg and transform into a string
    :param array: arg array, must be strings
    :return: a string
    """
    if array is None:
        return ""
    str = ""
    for i in range(len(array) - 1):
        str += array[i] + " "
    return str + array[-1]


def classify_mag(title, abstract, model_name="linearSVC", score=False):
    """
    simple function to make MAG classification
    :param title: Title of the paper to classify
    :param abstract: abstract of the paper to classify
    :param model_name: name of the model: linearSVC or SVC, default is linearSVC
    :param scores: bool, display score for all classes or not
    :return: the class or classes with score
    """
    specter = SentenceTransformer('allenai-specter')
    if model_name == "linearSVC":
        model = pickle.load(open("../models/linearSVC_model1.sav", 'rb'))
    elif model_name == "SVC":
        model = pickle.load(open("../models/SVC_model1.sav", 'rb'))
    else:
        path = "../models/" + model_name
        if isfile(path):
            model = pickle.load(open(path, 'rb'))
        else:
            raise ValueError("not a valid model name, try SVC or linearSVC or move model to models directory")

    embedding = specter.encode([title + 'SEP' + abstract])
    if score:
        scores = np.array(list(zip(model.decision_function(embedding)[0], mag_classes.values())), dtype=object)
        return scores[np.argsort(-scores[:, 0])]
    return mag_classes[model.predict(embedding)[0]]


def classify_mesh(title, abstract, model_name="linearSVC", score=False):
    """
    simple function to make MeSH classification
    :param title: Title of the paper to classify
    :param abstract: Abstract of the paper to classify
    :param model_name: name of the model: linearSVC or SVC
    :param classes: bool, display score for all classes or not
    :return: the class or classes with score
    """
    specter = SentenceTransformer('allenai-specter')
    if model_name == "linearSVC":
        model = pickle.load(open("../models/linearSVC_model1_mesh.sav", 'rb'))
    elif model_name == "SVC":
        model = pickle.load(open("../models/SVC_model1_mesh.sav", 'rb'))
    else:
        path = "../models/" + model_name
        if isfile(path):
            model = pickle.load(open(path, 'rb'))
        else:
            raise ValueError("not a valid model name, try SVC or linearSVC or move model to models directory")

    embedding = specter.encode([title + 'SEP' + abstract])
    if score:
        scores = np.array(list(zip(model.decision_function(embedding)[0], mesh_classes.values())), dtype=object)
        return scores[np.argsort(-scores[:, 0])]
    return mesh_classes[model.predict(embedding)[0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a scientific paper.")
    parser.add_argument("--type", help="Type of classification, MAG or MeSH", required=True)
    parser.add_argument("--model_name", help="Model name are (for now) linearSVC or SVC, linearSVC by default.")
    parser.add_argument("--title", nargs='*', help="Title of the document to classify.")
    parser.add_argument("--abstract", nargs='*', help="Abstract of the document to classify.")
    parser.add_argument("--scores", default=False, type=int, help="give score for each class, either 0 or 1")
    params = parser.parse_args()

    if params.type.lower() == "mag":
        classification = classify_mag(arg_to_string(params.title), arg_to_string(params.abstract), params.model_name, params.scores)
    elif params.type.lower() == "mesh":
        classification = classify_mesh(arg_to_string(params.title), arg_to_string(params.abstract), params.model_name, params.scores)
    else:
        raise ValueError("classification is either MAG or MeSH")

    print("\n", classification, "\n")