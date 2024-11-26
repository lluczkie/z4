import pickle

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""


def game_state_to_data_sample(game_state: dict):
    raise NotImplementedError()


if __name__ == "__main__":
    """ Example of how to read a pickled file, feel free to remove this"""
    with open(f"data/2024-10-14_12:52:37.pickle", 'rb') as f:
        data_file = pickle.load(f)
    print(data_file["block_size"])
    print(data_file["bounds"])
    print(data_file["data"])
