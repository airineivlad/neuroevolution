import multiprocessing
import os
import pickle

import numpy as np
import neat
import gym


env = gym.make("LunarLander-v2", new_step_api=True)
env.action_space.seed(42)


def fitness_function(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    env = gym.make("LunarLander-v2", new_step_api=True)
    env.action_space.seed(42)

    observation = env.reset()
    fitness = 0.0
    terminated = False

    while not terminated :
        nn_output = net.activate(observation)
        action = np.argmax(nn_output)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        if reward == 100:
            fitness += 200
        elif reward > 0:
            fitness += 1
        
        # Main engine
        if action == 2:
            # Penalize when using only
            reward -= 4

    fitnesses.append(fitness)

    return np.mean(fitnesses)


def run():

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), fitness_function)

    winner = pop.run(pe.evaluate)

    with open('models/lunar-lander_model', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    run()
