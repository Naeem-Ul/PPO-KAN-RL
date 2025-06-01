from kan_convolutional.kan_conv import KANConv2DLayer, KANConv1DLayer
from kan_convolutional.KANConv import KAN_Convolutional_Layer as CKAN
import torch 
import gymnasium as gym 
from torch import nn
from stable_baselines3 import PPO 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from kan import * 
from gymnasium import spaces
from custom_envs.half_cheetah_image_env import HalfCheetahObservationAndImageEnv

class CKAN_combination_input_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_image , features_dim: int = 256, input_shape=(1, 17)):
        print("observation_image: ", observation_image)
        observation_space = observation_image["observation"]
        image = observation_image["image"]
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        
        observation_n_input_channels = self.input_shape[0]
        image_n_input_channels = image.shape[0]
        

        print("input_shape: ", input_shape)
        print("n_input_channels: ", observation_n_input_channels)
        print("features_dim: ", features_dim)
        print("observation_space: ", observation_space)
        
        
        #image input
        self.image_cnn = nn.Sequential(
            KANConv2DLayer(input_dim=image_n_input_channels, output_dim=8, spline_order=3, grid_size=5, kernel_size=3, grid_range=[-1, 1], groups=1),
            KANConv2DLayer(input_dim=8, output_dim=16, spline_order=3, grid_size=5, kernel_size=3, grid_range=[-1, 1],groups=1),
            # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        
        #observation input
        self.observation_cnn = nn.Sequential(
            KANConv1DLayer(input_dim=observation_n_input_channels, output_dim=8, spline_order=3, grid_size=5, kernel_size=1, grid_range=[-1, 1], groups=1),
            KANConv1DLayer(input_dim=8, output_dim=16, spline_order=3, grid_size=5, kernel_size=1, grid_range=[-1, 1],groups=1),
            # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        
        #Concatenate the output of the two CNNs, the output is output of nn.Flatten()
        with torch.no_grad():
            n_flatten = self.image_cnn(
                torch.as_tensor(image.sample()[None]).float()
            ).shape[1] + self.observation_cnn(
                torch.as_tensor(observation_space.sample()[None]).unsqueeze(0).float()
            ).shape[1]
                
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())


    
    def preprocess_observation(self, state):
        state = state.unsqueeze(0)
        #The input shape is CxBxL where B is the batch size, C is the number of channels, and L is the length of the sequence.
        #Transpose the input shape to BxCxL
        state = state.transpose(0, 1) 
        # print("state.shape: ", state.shape)
        return state.to(self.device)
    
    def preprocess_image(self, image): 
        return image.to(self.device)
    
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observation = observations["observation"]
        image = observations["image"]
        
        observation = self.preprocess_observation(observation)
        image = self.preprocess_image(image)
        
        observation_features = self.observation_cnn(observation)
        image_features = self.image_cnn(image)
        
        features = torch.cat((observation_features, image_features), dim=1)
        
        return self.linear(features)


    
if __name__ == "__main__":
    import gymnasium as gym 
    policy_kwargs = dict(
        features_extractor_class=CKAN_combination_input_Extractor,
        features_extractor_kwargs=dict(features_dim=64, input_shape=(1 ,17)),
    )
    
    env = HalfCheetahObservationAndImageEnv(render_width=256, render_height=256)
    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
        
    print(model.policy)
    

    