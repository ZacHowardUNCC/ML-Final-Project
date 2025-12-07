#!/usr/bin/env python3
import json
from pathlib import Path


def parse_labels(labels_dir):

    labels_dir = Path(labels_dir).resolve(strict=True)

    # One object per photo, if photo has 3 of an object count is only incremented by 1. All unique objects counted.
    labels_dict = {
        "objects": {},
        "weather": {},
        "scene": {},
        "timeofday": {}
    }

    # Loop through all json files
    for file in labels_dir.glob("*.json"):
        label = json.loads(file.read_text())

        # Add unique objects to the labels dictionary
        unique_objects = set(object["category"] for object in label["frames"][0]["objects"]) # List comp loop
        for object in unique_objects:
            labels_dict["objects"][object] = labels_dict["objects"].get(object, 0) + 1

        # Get attributes per json file and add them to the labels dictionary
        attributes = label["attributes"]
        for attribute_name, attribute_value in attributes.items():
            labels_dict[attribute_name][attribute_value] = labels_dict[attribute_name].get(attribute_value, 0) + 1

    return labels_dict

def main():

    # Parse labels
    labels = parse_labels("data/labels")

    # Store labels dictionary in a json file
    output_file = Path("data/parsed_labels.json")
    with open(output_file, "w") as f:
        json.dump(labels, f, indent=2)
    print("parsed_labels.json file has been successfully created.")


if __name__ == "__main__":
    main()