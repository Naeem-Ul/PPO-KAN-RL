from kan_convolutional.kan_conv import KANConv1DLayer 
from kan_convolutional.KANConv import KAN_Convolutional_Layer as CKAN
import torch 
import gymnasium as gym 
from torch import nn
from stable_baselines3 import PPO 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from kan import * 
from gymnasium import spaces
from PPO_KAN import KAN_EXTRACTOR
import torchvision.transforms as T


class CKAN_state_input_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, input_shape=(1, 17), grid=5, k=3):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        #Number of stake timesteps: pre-specific 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        n_input_channels = self.input_shape[0]
        
        
        
        
        print("input_shape: ", input_shape)
        print("n_input_channels: ", n_input_channels)
        print("features_dim: ", features_dim)
        print("observation_space: ", observation_space)
        
        #torch kan conv
        self.cnn = nn.Sequential(
            KANConv1DLayer(input_dim=n_input_channels, output_dim=8, spline_order=k, grid_size=grid, kernel_size=1, grid_range=[-1, 1], groups=1),
            KANConv1DLayer(input_dim=8, output_dim=16, spline_order=k, grid_size=grid, kernel_size=1, grid_range=[-1, 1],groups=1),
            # CKAN(in_channels=5, out_channels=10, grid_size=5, kernel_size=(3,3)),
            # CKAN(in_channels=128, out_channels=256, grid_size=5, kernel_size=(3,3)),
            nn.Flatten(),
        )
        
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).unsqueeze(0).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    
    def preprocess(self, state):
        # transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.Resize(self.input_shape[1:]), T.ToTensor()])
        # print("state.shape: ", state.shape)
        # state = torch.from_numpy(state)
        
        state = state.unsqueeze(0)
        #The input shape is CxBxL where B is the batch size, C is the number of channels, and L is the length of the sequence.
        #Transpose the input shape to BxCxL
        state = state.transpose(0, 1) 
        # print("state.shape: ", state.shape)
        return state.to(self.device)



    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.preprocess(observations)
        # print("observation.shape ",observations.shape)
        return self.linear(self.cnn(observations))

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
        self.mlp_extractor = KAN_EXTRACTOR(self.features_dim, 
                                            last_layer_dim_pi=self.last_layer_dim_pi, 
                                            last_layer_dim_vf=self.last_layer_dim_vf,
                                            grid=self.grid, 
                                            k=self.k)
    
if __name__ == "__main__":
    import gymnasium as gym 
    policy_kwargs = dict(
        features_extractor_class=CKAN_state_input_Extractor,
        features_extractor_kwargs=dict(features_dim=64, input_shape=(1 ,17), grid=5, k=3),
        grid=1, 
        k=3,
    )
    
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    model = PPO(PPO_KAN, env, verbose=1, policy_kwargs=policy_kwargs)
        
    print(model.policy)
    