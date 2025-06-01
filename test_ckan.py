from kan_convolutional.KANConv import KAN_Convolutional_Layer as CKAN 
import torch 
import gymnasium as gym 
import time
from torch import nn
from stable_baselines3 import PPO 
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from kan import * 
import torchvision.transforms as T
from gymnasium import spaces
from custom_envs.half_cheetah_image_env import HalfCheetahImageEnv

class CKAN_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        
        self.cnn = nn.Sequential(
            CKAN(in_channels=n_input_channels, out_channels=3, grid_size=5, kernel_size=(3,3)),
            CKAN(in_channels=3, out_channels=5, grid_size=5, kernel_size=(3,3)),
            CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class KAN_Head(nn.Module): 
    def __init__(
            self, 
            feature_dim, 
            last_layer_dim_pi, 
            last_layer_dim_vf,
            grid, 
            k,
        ):
        super().__init__()
        print("feature_dim: ", feature_dim)
        print("last_layer_dim_pi: ", last_layer_dim_pi)
        print("last_layer_dim_vf: ", last_layer_dim_vf)
        print("grid: ", grid)
        print("k: ", k)
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Policy network
        self.policy_net = nn.Sequential(
            CKAN(in_channels=feature_dim, out_channels=64, grid_size=grid, kernel_size=(3,3))
        )
        # Value network
        self.value_net = nn.Sequential(
            CKAN(in_channels=feature_dim, out_channels=1, grid_size=grid, kernel_size=(3,3))
        )
        
    def forward(self, features):
        """
            :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)
        
class PPO_KAN(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            grid, 
            k,
            *args,
            **kwargs,
        ):
        self.grid = grid
        self.k = k
        self.last_layer_dim_pi = action_space.shape[0]
        self.last_layer_dim_vf = 1
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = KAN_Head(self.features_dim, 
                                            last_layer_dim_pi=self.last_layer_dim_pi, 
                                            last_layer_dim_vf=self.last_layer_dim_vf,
                                            grid=self.grid, 
                                            k=self.k)
        
        
        
if __name__ == "__main__":
    import gymnasium as gym 
    policy_kwargs = dict(
        features_extractor_class=CKAN_Extractor,
        features_extractor_kwargs=dict(features_dim=64),
    )
    
    env = HalfCheetahImageEnv(render_width=256, render_height=256)
    model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
        
    print(model.policy)
    