
# import torch
from PPO_CKAN_state_input import CKAN_state_input_Extractor

hyperparams = {
    "HalfCheetah-v5": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,17)),
        )
    ),
    "Ant-v5": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,105)),
        )
    ),
    "Hopper-v5": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,11)),
        )
    ),
    "InvertedPendulum-v5": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,4)),
        )
    ),
    "InvertedDoublePendulum-v5": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,4)),
        )
    ),
    "Swimmer-v5": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,8)),
        )
    ),
    
    
    
    # V-4 
    "HalfCheetah-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,17)),
        )
    ),
    "Ant-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,105)),
        )
    ),
    "Hopper-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,11)),
        )
    ),
    "InvertedPendulum-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,4)),
        )
    ),
    "InvertedDoublePendulum-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,4)),
        )
    ),
    "Swimmer-v4": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MlpPolicy",
        batch_size=1_000,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,8)),
        )
    )
}
