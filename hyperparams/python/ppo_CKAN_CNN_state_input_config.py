
# import torch
from PPO_CKAN_CNN_state_input import CKAN_CNN_state_input_Extractor 

hyperparams = {
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
                    features_extractor_class=CKAN_CNN_state_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,17)),
        )
    )
}
