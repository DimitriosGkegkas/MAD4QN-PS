from train.MultiAgentTrainer import MultiAgentTrainerParallel, TrainerConfig


if __name__ == '__main__':
    config = TrainerConfig(
        algorithm_identifier="test",
        batch_size=2,
        num_env=3,
        evaluation_step= 2,
        max_train_steps= 100, # Maximum steps per episode, can be adjusted based on the environment
        max_evaluation_steps=20,
        
        # parallel=False,
        # envision=True,
        # evaluate=True
    )

    trainer = MultiAgentTrainerParallel(config)
    trainer.preload("models/test/agent_checkpoint.pth")
    

    trainer.train()
    # trainer.envision(1)
    