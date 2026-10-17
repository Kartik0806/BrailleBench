import json

def get_data(label_file):
    data = []

    with open(label_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    print(data[:2])

    return data
