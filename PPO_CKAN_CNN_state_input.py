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

class CKAN_CNN_state_input_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box , features_dim: int = 256, input_shape=(1, 17)):
       
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        n_input_channels = self.input_shape[0]
        
        

        print("input_shape: ", input_shape)
        print("n_input_channels: ", n_input_channels)
        print("features_dim: ", features_dim)
        print("observation_space: ", observation_space)
        
        
        #observation input cnn network
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1),
            nn.ReLU(),
            # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        
        #observation input ckan network
        self.ckan = nn.Sequential(
            KANConv1DLayer(input_dim=n_input_channels, output_dim=8, spline_order=3, grid_size=5, kernel_size=1, grid_range=[-1, 1], groups=1),
            KANConv1DLayer(input_dim=8, output_dim=16, spline_order=3, grid_size=5, kernel_size=1, grid_range=[-1, 1],groups=1),
            # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        
        #Concatenate the output of the two CNNs, the output is output of nn.Flatten()

        
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).unsqueeze(0).float()
            ).shape[1] + self.ckan(
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
    
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.preprocess_observation(observations)
        
        ckan_features = self.ckan(observations)
        cnn_features = self.cnn(observations)
        
        
        features = torch.cat((ckan_features, cnn_features), dim=1)
        
        return self.linear(features)


    
if __name__ == "__main__":
    import gymnasium as gym 
    policy_kwargs = dict(
        features_extractor_class=CKAN_CNN_state_input_Extractor,
        features_extractor_kwargs=dict(features_dim=64, input_shape=(1 ,17)),
    )
    
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
        
    print(model.policy)
    

    