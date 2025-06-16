from Agent.agent import AgentConfig
from train.MultiAgentTrainer import MultiAgentTrainerParallel, TrainerConfig


if __name__ == '__main__':
    config = TrainerConfig(
        algorithm_identifier="SAC2",
        num_env=2,
        evaluation_step=10,
        max_training_steps=500, # Maximum steps per episode, can be adjusted based on the environment
        max_evaluation_steps=500,
        
        scenario_subdir = "scenarios/sumo/multi_scenario_part",
        observation_shape=(128, 128, 3),  # Shape of the observation space, can be adjusted based on the environment
        
        # parallel=False,
        # envision=True,
        # evaluate=True
    )
    agent_config = AgentConfig(
        feature_dim=64,
        message_dim=4,
        batch_size= 256, 
        
        communication_hidden_dim= [32, 16],
        critic_hidden_dim= [128, 128, 32],
        actor_hidden_dim= [128, 128, 32],
        memory_max_size= int(1000),
        
        reconstruction_coef=0.1,
        img_reconstruction_coef=0.1,
        reg_coef=0.1,
        smoothness_coef= 0.01,
    )

    trainer = MultiAgentTrainerParallel(config, agent_config)
    # trainer.preload("models/agent_checkpoint.pth")
    

    trainer.train()
    # trainer.envision(1)
    