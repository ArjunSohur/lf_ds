# Library imports
import json

# Variables
input_file_path = "yelp_dataset/yelp_academic_dataset_review.json"
output_file_path = "manage_dataset/truncated_data.json"


def extract(datapoints):
    with open(input_file_path, "r") as input_file, open(
        output_file_path, "w"
    ) as output_file:
        lines_copied = 0
        print("Loaded review dataset")

        for line in input_file:
            data = json.loads(line)

            json.dump(data, output_file)
            output_file.write("\n")

            lines_copied += 1

            if lines_copied >= datapoints:
                break

    print(f"{datapoints} lines written to {output_file_path}.")


def get_data():
    list_data = []

    with open(output_file_path, "r") as out_file:
        for line in out_file:
            temp = json.loads(line)
            list_data.append(temp)

    return list_data
