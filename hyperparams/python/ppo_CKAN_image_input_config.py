
# import torch
from PPO_CKAN_image_input import CKAN_image_input_Extractor

hyperparams = {
    "HalfCheetahImage-v0": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="CnnPolicy",
        batch_size=8,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_image_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64),
        )
    )
}
