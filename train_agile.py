# Taken from https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_ppo_tutorial.html#ppo-tutorial

import os

import imageio
import gymnasium as gym
import numpy as np
import torch
from agilerl.algorithms.ppo import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
from TradingEnv_agile import TradingEnv
import StockData
from gymnasium.envs.registration import register

register(
    id='TradingEnv',  # Unique identifier for your environment
    entry_point='TradingEnv_agile:TradingEnv',  # Module and class of your environment
)

if __name__ == "__main__":

    # Initial hyperparameters
    INIT_HP = {
        "POP_SIZE": 8,  # Population size
        "DISCRETE_ACTIONS": False,  # Discrete action space
        "BATCH_SIZE": 128,  # Batch size
        "LR": 0.001,  # Learning rate
        "LEARN_STEP": 1024,  # Learning frequency
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "UPDATE_EPOCHS": 8,  # Number of policy update epochs
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,  # Use with RGB states
        "TARGET_SCORE": 400.0,  # Target score that will beat the environment
        "MAX_STEPS": 1500000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 100000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 3,  # Number of evaluation episodes
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
    }

    # Mutation parameters
    MUT_P = {
        # Mutation probabilities
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0.2,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        # Learning HPs to choose from
        "RL_HP_SELECTION": ["lr", "batch_size", "learn_step"],
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 42,  # Random seed
        # Define max and min limits for mutating RL hyperparams
        "MIN_LR": 0.0001,
        "MAX_LR": 0.01,
        "MIN_BATCH_SIZE": 8,
        "MAX_BATCH_SIZE": 1024,
        "MIN_LEARN_STEP": 256,
        "MAX_LEARN_STEP": 8192,
    }

    num_envs=8
    env = make_vect_envs("TradingEnv", num_envs=num_envs)  # Create environment

    try:
        state_dim = env.single_observation_space.n  # For discrete observation spaces
        one_hot = True
    except AttributeError:
        state_dim = env.single_observation_space.shape  # For continuous observation spaces
        one_hot = False

    try:
        action_dim = env.single_action_space.n  # For discrete action spaces
    except AttributeError:
        action_dim = env.single_action_space.shape[0]  # For continuous action spaces

    if INIT_HP["CHANNELS_LAST"]:
        # Adjust dimensions for PyTorch API (C, H, W), for envs with RGB image states
        state_dim = (state_dim[2], state_dim[0], state_dim[1])


    # Set-up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}

    # Define a population
    pop = create_population(
        algo="PPO",  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=net_config,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameter
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,
        device=device,
    )

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )

    mutations = Mutations(
        algo="PPO",
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        rl_hp_selection=MUT_P["RL_HP_SELECTION"],
        min_lr=MUT_P["MIN_LR"],
        max_lr=MUT_P["MAX_LR"],
        min_batch_size=MUT_P["MAX_BATCH_SIZE"],
        max_batch_size=MUT_P["MAX_BATCH_SIZE"],
        min_learn_step=MUT_P["MIN_LEARN_STEP"],
        max_learn_step=MUT_P["MAX_LEARN_STEP"],
        mutation_sd=MUT_P["MUT_SD"],
        arch=net_config["arch"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )


    # Define a save path for our trained agent
    save_path = "models/agile/PPO_trained_agent 2.pt"

    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name="TradingEnv",
        algo="PPO",
        pop=pop,
        INIT_HP=INIT_HP,
        MUT_P=MUT_P,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        tournament=tournament,
        mutation=mutations,
        wb=False,  # Boolean flag to record run with Weights & Biases
        save_elite=True,  # Boolean flag to save the elite agent in the population
        elite_path=save_path,
    )