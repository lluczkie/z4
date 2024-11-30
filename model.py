import pickle
import numpy as np
from snake import Direction
from math import log
"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""

def game_state_to_data_sample(game_state: dict, block_size: int, bounds: tuple):
    snake_body = game_state["snake_body"]
    head = snake_body[-1]
    food = game_state["food"]

    is_wall_left = True if head[0] == 0 else False
    is_wall_right = True if head[0] == (bounds[0]-block_size)/block_size else False
    is_wall_up = True if head[1] == 0 else False
    is_wall_down = True if head[1] == (bounds[1]-block_size)/block_size else False
    
    snake_module_left = (head[0]-block_size, head[1])
    snake_module_right = (head[0]+block_size, head[1])
    snake_module_up = (head[0], head[1]-block_size)
    snake_module_down = (head[0], head[1]+block_size)
    
    is_snake_left = True if snake_module_left in snake_body else False
    is_snake_right = True if snake_module_right in snake_body else False
    is_snake_up = True if snake_module_up in snake_body else False
    is_snake_down = True if snake_module_down in snake_body else False
    
    is_food_left = True if head[1] == food[1] and head[0] == food[0] + block_size else False
    is_food_right = True if head[1] == food[1] and head[0] + block_size == food[0] else False
    is_food_up = True if head[0] == food[0] and head[1] == food[1] + block_size else False
    is_food_down = True if head[0] == food[0] and head[1] + block_size == food[1] else False

    is_obstacle_left = True if is_wall_left or is_snake_left else False
    is_obstacle_right = True if is_wall_right or is_snake_right else False
    is_obstacle_up = True if is_wall_up or is_snake_up else False
    is_obstacle_down = True if is_wall_down or is_snake_down else False

    data_sample = np.array([[is_obstacle_left, is_obstacle_right, is_obstacle_up, is_obstacle_down, is_food_left, is_food_right, is_food_up, is_food_down]])

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
    return game_states, np.array(directions)

def create_training_data(states, block_size, bounds):
    training_data = np.array([[0, 1, 2, 3, 4, 5, 6, 7]]) 
    for state in states:
        attributes = game_state_to_data_sample(state, block_size, bounds)
        training_data = np.concatenate((training_data, attributes), axis=0)
    return training_data

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
    mask = data_to_divide[:, attribute_index[0][0]] == True
    anti_mask = np.logical_not(mask)
    attribute_mask = data_to_divide[0, :] != attribute_id
    data_to_divide = data_to_divide[:, attribute_mask]
    sub_collection_true = data_to_divide[1:, :][mask]
    directions_true = directions[mask]
    sub_collection_false = data_to_divide[1:, :][anti_mask]
    directions_false = directions[anti_mask]
    return [(sub_collection_true, directions_true), (sub_collection_false, directions_false)]

def inf_gain(data_to_divide, directions, attribute_id):
    [(_, dirs1), (_, dirs2)] = divide_by_attribute(data_to_divide, directions, attribute_id)
    inf = (dirs1.size*get_entropy(dirs1) + dirs2.size*get_entropy(dirs2))/directions.size
    return get_entropy(directions)-inf

def ID3(training_data, directions):
    if np.all(directions == directions[0]):
        return directions[0]
    if training_data.size == 0:
        counts = np.bincount(directions)
        return np.argmax(counts)
    
    attributes = training_data[0, :]
    gains = [inf_gain(training_data, directions, attr) for attr in attributes]
    divisor = attributes[np.argmax(gains)]
    [(attr_true, dir_true), (attr_false, dir_false)] = divide_by_attribute(training_data, directions, divisor)
    return {True: ID3(attr_true, dir_true), False: ID3(attr_false, dir_false)}

if __name__ == "__main__":
    states, directions = get_states_and_directions_from_pickle("data/2024-11-30_17:25:59.pickle")
    training_data = create_training_data(states, 30, (300, 300))
    test_divide = np.array([[0, 1,3,4], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    test_dirs = np.array([Direction.DOWN, Direction.LEFT, Direction.UP, Direction.LEFT])
    
    ID3(test_divide, test_dirs)