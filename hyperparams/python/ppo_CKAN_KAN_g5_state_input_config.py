
# import torch
from PPO_CKAN_KAN_state_input import CKAN_state_input_Extractor
from PPO_KAN import PPO_KAN

hyperparams = {
    # V-4 
    "HalfCheetah-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="PPO_KAN.PPO_KAN",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,17)),
                    grid=5, # grid for KAN 
                    k=3, # k for KAN
        )
    ),
    "Ant-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="PPO_KAN.PPO_KAN",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,105)),
                    grid=5, # grid for KAN 
                    k=3, # k for KAN
        )
    ),
    "Hopper-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="PPO_KAN.PPO_KAN",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,11)),
                    grid=5, # grid for KAN 
                    k=3, # k for KAN
        )
    ),
    "InvertedPendulum-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="PPO_KAN.PPO_KAN",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,4)),
                    grid=5, # grid for KAN 
                    k=3, # k for KAN
        )
    ),
    "InvertedDoublePendulum-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="PPO_KAN.PPO_KAN",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,4)),
                    grid=5, # grid for KAN 
                    k=3, # k for KAN
        )
    ),
    "Swimmer-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="PPO_KAN.PPO_KAN",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,8)),
                    grid=5, # grid for KAN 
                    k=3, # k for KAN
        )
    )
}
