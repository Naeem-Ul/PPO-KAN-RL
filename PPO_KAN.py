from torch import nn
import torch
from stable_baselines3 import PPO 
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from kan import * 


class KAN_LAYER(nn.Module):
    def __init__(self, input_size, output_size, grid, k): 
        super(KAN_LAYER, self).__init__()
        self.input_size = input_size    
        self.output_size = output_size
        self.grid = grid
        self.k = k
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KANLayer(in_dim=self.input_size, out_dim=self.output_size, num=self.grid, k=self.k, device=self.device)
    
    def forward(self, x): 
        output, preacts, postacts, postspline = self.model(x)
        return output

class KAN_EXTRACTOR(nn.Module): 
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
            KAN_LAYER(input_size=feature_dim, output_size=64, grid=grid, k=k),
            KAN_LAYER(input_size=64, output_size=64, grid=grid, k=k),
            KAN_LAYER(input_size=64, output_size=64, grid=grid, k=k),
            KAN_LAYER(input_size=64, output_size=last_layer_dim_pi, grid=grid, k=k),
        )
        # Value network
        self.value_net = nn.Sequential(
            KAN_LAYER(input_size=feature_dim, output_size=64, grid=grid, k=k),
            KAN_LAYER(input_size=64, output_size=64, grid=grid, k=k),
            KAN_LAYER(input_size=64, output_size=64, grid=grid, k=k),
            KAN_LAYER(input_size=64, output_size=last_layer_dim_vf, grid=grid, k=k),
        )
        
    def forward(self, features):
        """
            :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
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
        self.mlp_extractor = KAN_EXTRACTOR(self.features_dim, 
                                            last_layer_dim_pi=self.last_layer_dim_pi, 
                                            last_layer_dim_vf=self.last_layer_dim_vf,
                                            grid=self.grid, 
                                            k=self.k)
        
        
        
if __name__ == "__main__":
    import gymnasium as gym 
    policy_kwargs = dict(
        grid=10, 
        k=3,
    )
    
    
    # env = gym.make("Ant-v4", render_mode="rgb_array")
    model = PPO(PPO_KAN, 'Ant-v4', verbose=1, policy_kwargs=policy_kwargs)
    # model = PPO("MlpPolicy", 'CartPole-v1', verbose=1,policy_kwargs=policy_kwargs)
        
    print(model.policy)
    