from agent import AgentConfig
from train.MultiAgentTrainer import MultiAgentTrainer, TrainerConfig


if __name__ == '__main__':
    config = TrainerConfig(
        algorithm_identifier="SAC3",
        num_env=2,
        evaluation_step=10,
        max_training_steps=400, # Maximum steps per episode, can be adjusted based on the environment
        max_evaluation_steps=200,
        agent_count=4,
        observation_shape=(128, 128, 3),  # Shape of the observation space, can be adjusted based on the environment
        eval_scenarios=[0,1, 192, 195],
        # parallel=False,
        # envision=True,
        evaluate=True
    )
    agent_config = AgentConfig(
        feature_dim=64,
        message_dim=8,
        batch_size=258, 
        
        communication_hidden_dim= [32, 16],
        critic_hidden_dim= [128, 128, 32],
        actor_hidden_dim= [128, 128, 32],
        memory_max_size= int(100000),
        
        reconstruction_coef=0.1,
        img_reconstruction_coef=0.1,
        reg_coef=0.1,
        smoothness_coef= 0.01,
    )

    trainer = MultiAgentTrainer(config, agent_config)
    # trainer.preload("models/SAC2/20250619/best_checkpoint.pth")
    

    # trainer.train()
    trainer.evaluate()
    # trainer.envision(195)
    