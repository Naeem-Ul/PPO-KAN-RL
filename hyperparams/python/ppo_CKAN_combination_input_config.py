
# import torch
from PPO_CKAN_combination import CKAN_combination_input_Extractor 

hyperparams = {
    "HalfCheetahObservationAndImage-v0": dict(
        normalize=True,
        n_envs=1,
        n_timesteps=10,
        policy="MultiInputPolicy",
        batch_size=8,
        n_steps=1_000,
        gamma=0.98,
        learning_rate=0.0003,
        n_epochs=5,
        policy_kwargs = dict(
                    features_extractor_class=CKAN_combination_input_Extractor,
                    features_extractor_kwargs=dict(features_dim=64, input_shape=(1,17)),
        )
    )
}
