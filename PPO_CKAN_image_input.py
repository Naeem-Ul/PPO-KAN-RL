from kan_convolutional.kan_conv import KANConv2DLayer 
from kan_convolutional.KANConv import KAN_Convolutional_Layer as CKAN
import torch 
import gymnasium as gym 
from torch import nn
from stable_baselines3 import PPO 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from kan import * 
from gymnasium import spaces
from custom_envs.half_cheetah_image_env import HalfCheetahImageEnv
import torchvision.transforms as T

class CKAN_image_input_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Image shape: ", observation_space.shape)
        n_input_channels = observation_space.shape[0]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        print("n_input_channels: ", n_input_channels)
        print("features_dim: ", features_dim)
        print("observation_space: ", observation_space)

        #torch kan conv
        self.cnn = nn.Sequential(
            KANConv2DLayer(input_dim=n_input_channels, output_dim=8, spline_order=3, grid_size=5, kernel_size=3, grid_range=[-1, 1], groups=1),
            KANConv2DLayer(input_dim=8, output_dim=16, spline_order=3, grid_size=5, kernel_size=3, grid_range=[-1, 1],groups=1),
            # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        
        #kan conv
        # self.cnn = nn.Sequential(
        #     CKAN(in_channels=n_input_channels, out_channels=3, grid_size=5, kernel_size=(1,1)),
        #     CKAN(in_channels=3, out_channels=6, grid_size=5, kernel_size=(1,1)),
        #     # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
        #     # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
        #     nn.Flatten(),
        # )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def preprocess_image(self, image):
        # print("image value range: ", torch.min(image), torch.max(image))
        return image.to(self.device)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.preprocess_image(observations)
        return self.linear(self.cnn(observations))


    
if __name__ == "__main__":
    import gymnasium as gym 
    policy_kwargs = dict(
        features_extractor_class=CKAN_image_input_Extractor,
        features_extractor_kwargs=dict(features_dim=64),
    )
    
    env = HalfCheetahImageEnv(render_width=256, render_height=256)
    model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    

    
    
    print(model.policy)
    