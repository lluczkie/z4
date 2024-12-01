import pickle
import numpy as np
from snake import Direction
from math import log
import json
from  sklearn.metrics import precision_score
"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""

def combine_pickles():
    run=[]
    open('data/merged.pickle', 'w').close()
    for i in range(10):
        with open(f'data/run{i}.pickle', 'rb') as run_file:
            run = pickle.load(run_file)
        with open("data/merged.pickle", "ab") as merged_file:
            pickle.dump(run, merged_file)

def game_state_to_data_sample(game_state: dict, block_size: int, bounds: tuple):
    snake_body = game_state["snake_body"]
    head = snake_body[-1]
    food = game_state["food"]
    snake_direction = game_state["snake_direction"]

    is_wall_left = True if head[0] == 0 else False
    is_wall_right = True if head[0] == (bounds[0]-block_size) else False
    is_wall_up = True if head[1] == 0 else False
    is_wall_down = True if head[1] == (bounds[1]-block_size) else False
    
    snake_module_left = (head[0] - block_size, head[1])
    snake_module_right = (head[0] + block_size, head[1])
    snake_module_up = (head[0], head[1] - block_size)
    snake_module_down = (head[0], head[1] + block_size)
    
    is_snake_left = True if snake_module_left in snake_body else False
    is_snake_right = True if snake_module_right in snake_body else False
    is_snake_up = True if snake_module_up in snake_body else False
    is_snake_down = True if snake_module_down in snake_body else False
    
    is_obstacle_left = True if is_wall_left or is_snake_left else False
    is_obstacle_right = True if is_wall_right or is_snake_right else False
    is_obstacle_up = True if is_wall_up or is_snake_up else False
    is_obstacle_down = True if is_wall_down or is_snake_down else False

    is_food_in_snake_direction = False
    if head[1] == food[1] and head[0] < food[0] and snake_direction == Direction.LEFT:
        is_food_in_snake_direction = True
    if head[1] == food[1] and head[0] > food[0] and snake_direction == Direction.RIGHT:
        is_food_in_snake_direction = True
    if head[0] == food[0] and head[1] > food[1] and snake_direction == Direction.UP:
        is_food_in_snake_direction = True
    if head[0] == food[0] and head[1] < food[1] and snake_direction == Direction.DOWN:
        is_food_in_snake_direction = True

    is_food_left = True if head[1] == food[1] and head[0] > food[0] else False
    is_food_right = True if head[1] == food[1] and head[0] < food[0] else False
    is_food_up = True if head[0] == food[0] and head[1] > food[1] else False
    is_food_down = True if head[0] == food[0] and head[1] < food[1] else False

    data_sample = np.array([[is_obstacle_left, is_obstacle_right, is_obstacle_up, is_obstacle_down, is_food_left, is_food_right, is_food_up, is_food_down, is_food_in_snake_direction]])

    return data_sample

def get_states_and_directions_from_pickle(filename):
    game_states = []
    directions = []
    data_file = []
    with open(filename, 'rb') as f:
        data_file = pickle.load(f)
    for data_entry in data_file["data"]:
        game_states.append(data_entry[0])
        directions.append(data_entry[1])
    return game_states, directions

def process_data(states, directions, block_size, bounds):
    processed_data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]) 
    new_dirs = []
    for state, dir in zip(states, directions):
        attributes = game_state_to_data_sample(state, block_size, bounds)

        # never collide
        if attributes[0][0] and dir == Direction.LEFT:
            continue
        if attributes[0][1] and dir == Direction.RIGHT:
            continue
        if attributes[0][2] and dir == Direction.UP:
            continue
        if attributes[0][3] and dir == Direction.DOWN:
            continue

        # always go for food if possible
        if attributes[0][4] and not attributes[0][0]:
            dir = Direction.LEFT
        if attributes[0][5] and not attributes[0][1]:
            dir = Direction.RIGHT
        if attributes[0][6] and not attributes[0][2]:
            dir = Direction.UP
        if attributes[0][7] and not attributes[0][3]:
            dir = Direction.DOWN
        
        new_dirs.append(dir)
        processed_data = np.concatenate((processed_data, attributes), axis=0)
        
    return processed_data, np.array(new_dirs)

def get_entropy(directions):
    entropy = 0
    size = len(directions)
    if size == 0:
        return entropy
    frequencies = np.bincount(directions, minlength=4)/size
    for dir in Direction:
        if frequencies[dir] > 0:
            entropy -= frequencies[dir]*log(frequencies[dir])
    return entropy

def divide_by_attribute(data_to_divide, directions, attribute_id):
    attribute_index = np.where(data_to_divide[0, :] == attribute_id)
    mask = data_to_divide[1:, attribute_index[0][0]] == True
    anti_mask = np.logical_not(mask)
    attribute_mask = data_to_divide[0, :] != attribute_id
    data_to_divide = data_to_divide[:, attribute_mask]
    sub_collection_true = np.concatenate([data_to_divide[0:1, :], data_to_divide[1:, :][mask]], axis=0)
    directions_true = directions[mask]
    sub_collection_false = np.concatenate([data_to_divide[0:1, :], data_to_divide[1:, :][anti_mask]])
    directions_false = directions[anti_mask]
    return [(sub_collection_true, directions_true), (sub_collection_false, directions_false)]

def inf_gain(data_to_divide, directions, attribute_id):
    [(_, dirs1), (_, dirs2)] = divide_by_attribute(data_to_divide, directions, attribute_id)
    inf = (dirs1.size*get_entropy(dirs1) + dirs2.size*get_entropy(dirs2))/directions.size
    return get_entropy(directions)-inf

def ID3(training_data, directions):
    if directions.size == 0:
        return str(np.random.randint(0, 4))
    if np.all(directions == directions[0]):
        return str(directions[0])
    if training_data.size == 0:
        counts = np.bincount(directions)
        return str(np.argmax(counts))
    
    attributes = training_data[0, :]
    gains = [inf_gain(training_data, directions, attr) for attr in attributes]
    divisor = attributes[np.argmax(gains)]
    [(attr_true, dir_true), (attr_false, dir_false)] = divide_by_attribute(training_data, directions, divisor)
    return {str(divisor): {True: ID3(attr_true, dir_true), False: ID3(attr_false, dir_false)}}

def split_data(data, dirs, percent):
    # mix samples from different runs
    permutation = np.random.permutation(data.shape[0] - 1)
    np.take(data[1:, :], permutation, axis=0, out=data[1:, :])
    np.take(dirs, permutation, out=dirs)
    # take 0-100 percent of data
    perc = (data.shape[0]-1)*percent // 100
    data = data[:perc+1]
    dirs = dirs[:perc]
    # split
    split_ix = data.shape[0] // 5
    train_data = data[:4*split_ix+1, :]
    test_data = data[4*split_ix+1:, :]
    train_dirs = dirs[:4*split_ix]
    test_dirs = dirs[4*split_ix:]

    return  (train_data, train_dirs), (test_data, test_dirs)

def act_from_data_sample(id3, data_sample):
    level = list(id3.keys())[0]
    next = id3[level][str(bool(data_sample[int(level)])).lower()]
    while type(next) == dict:
        level = list(next.keys())[0]
        next=next[level][str(bool(data_sample[int(level)])).lower()]
    return int(next)

if __name__ == "__main__":
    combine_pickles()
    states, directions = get_states_and_directions_from_pickle(f"data/merged.pickle")
    processed_data, processed_directions = process_data(states, directions, 30, (300, 300))
    (train_data, train_dirs), (test_data, test_dirs) = split_data(processed_data, processed_directions, 100)
    tree = ID3(train_data, train_dirs)
    out_file = open(f"tree.json", "w")
    json.dump(tree, out_file, indent = 2)
    out_file.close()

    f = open('tree.json', 'r')
    id3 = json.load(f)
    f.close()

    pred_dirs = []
    for sample in test_data:
        pred_dirs.append(act_from_data_sample(id3, sample))
    ps = precision_score(test_dirs, pred_dirs, average=None)
    pass
