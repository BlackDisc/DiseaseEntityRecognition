import argparse
import json
from flair.nn import Classifier
from flair.data import Sentence
from nervaluate import Evaluator
from tqdm import tqdm



def parse_input_data(path):
    """ Proces input data in predefined format to list of dicts format

    Args:
        path: path to text file with input data with predifined format

    Returns:
        list of dicts with keys: 
           "id" - identification number of paper
           "text" - merged title and abstract
           "entities" - list of recognized entities 
    """
    with open(path, "r") as f:
        lines = f.readlines()

        current_id = None
        texts_dict = {}
        disease_dict = {}

        for line in lines:
            if line[0] == "\n":
                current_id = None
                continue
            if current_id is None:
                current_id = line.split("|")[0]
                texts_dict[current_id] = {}
                disease_dict[current_id] = []

            if len(line.split("|")) > 2:
                texts_dict[current_id][line.split("|")[1]] = line.split("|")[2]
            elif len(line.split("\t")) > 5:
                disease_dict[current_id].append(tuple(line.split("\t")[1:]))

        parsed_text = [
            {
                "id": key,
                "text": texts_dict[key]["t"].strip()
                + " "
                + texts_dict[key]["a"].strip(),
                "entities": [
                    (int(ent[0]), int(ent[1]), "Disease") for ent in disease_dict[key]
                ],
            }
            for key in texts_dict.keys()
        ]

    return parsed_text


def predict_ner(tagger, data):
    """Recognize entities for data in list of dicts format 

    Args:
        tagger : Model from flair framework used to NEr recognition
        data : data in list of dicts format

    Returns:
        predictions in list of dicts format
    """
    results = []
    for text in tqdm(data):
        sentence = Sentence(text["text"])
        tagger.predict(sentence)

        ner_list = []
        for span in sentence.get_spans():
            if span.tag == "Disease":
                ner_list.append((span.start_position, span.end_position, span.tag))

        results.append({"id": text["id"], "text": text["text"], "entities": ner_list})

    return results


def get_iob_annotation(text):
    """Prepare annotation in IOB format

    Args:
        text : dict from list of dicts. Represents one paper

    Returns:
        annotations in IOB format (list of tags for each token)
    """    
    text_processed  = Sentence(text['text'])

    text['entities'].sort(key=lambda ent: ent[0])

    entity_iterator = iter(text['entities'])
    current_entity = next(entity_iterator, None)
    iob_annotations = []
    for token in text_processed:
        if current_entity is not None:
            while token.start_position > int(current_entity[1]):
                current_entity = next(entity_iterator, None)
                if current_entity is None:
                    break
        if current_entity is not None:
            # Entity starts after token 
            if token.start_position < int(current_entity[0]):
                iob_annotations.append('O')
                continue
            # Start of entity and token aligned
            elif token.start_position == int(current_entity[0]):
                iob_annotations.append('B-Disease')
                
                # One token entity
                if token.end_position >= int(current_entity[1]):
                    current_entity = next(entity_iterator, None)
                    
            # Entity started before token 
            else:
                iob_annotations.append('I-Disease')
                
                # Last token of entity
                if token.end_position == int(current_entity[1]):
                    current_entity = next(entity_iterator, None)
        else:
            iob_annotations.append('O')
            
    return iob_annotations


def eval_iob_predictions(predictions_data, gt_data):
    """Evaluation of NER

    Args:
        predictions_data : predition data in list of dicts format
        gt_data : ground truth annotations in list of dicts format

    """
    gt_annotations = [get_iob_annotation(gt) for gt in gt_data]
    predicted_annotations = [get_iob_annotation(prediction) for prediction in predictions_data]
    evaluator = Evaluator(gt_annotations,
                          predicted_annotations,
                          tags=['Disease'],
                          loader="list")
    
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    print('Results of evaluation ')
    print(results['ent_type'])


def save_predictions(data, output_path):
    """Saving predictions in json format

    Args:
        data : list of dicts format
        output_path : path to output json file
    """
    with open(output_path, 'w') as f:
        f.write(json.dumps(data))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Disease Entity Recognition (DER)",
        description="Recognize Disease Entities in text in provided format",
    )
    parser.add_argument('--input_path', required=True,  type=str, help='path to input file')
    parser.add_argument('--output', default='output.json',  type=str, help='output file name. Default: output.json')
    args = parser.parse_args()

    
    # Params
    data_path = args.input_path

    # Parsing input data
    parsed_data = parse_input_data(data_path)
    # Loading ner model
    ner_tagger = Classifier.load("hunflair2")
    # Entity reconition 
    ner_predictions = predict_ner(ner_tagger, parsed_data)
    # Saving predictions
    save_predictions(ner_predictions, args.output)
    # Evaluate predictions
    eval_iob_predictions(ner_predictions, parsed_data)





