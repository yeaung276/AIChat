import os


def load_dataset(path):
    txt_file = [f for f in os.listdir(path) if f.endswith(".txt")][0]

    dataset = []
    with open(os.path.join(path, txt_file)) as f:
        for line in f:
            file_id, transcript = line.strip().split(" ", 1)
            dataset.append((os.path.join(path, file_id + ".flac"), transcript))

    return dataset
