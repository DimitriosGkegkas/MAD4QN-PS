from train.MultiAgentTrainer import MultiAgentTrainer
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Quiet noisy loggers
logging.getLogger("SMARTS").setLevel(logging.WARNING)
logging.getLogger("SensorManager").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("Client").setLevel(logging.WARNING)

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):    
    
    trainer = MultiAgentTrainer(cfg)

    # Preload checkpoint if specified
    if cfg.get("preload"):
        print(f"Preloading checkpoint: {cfg.preload}")
        trainer.preload(cfg.preload)

    # Dispatch based on mode
    if cfg.mode.action == "train":
        trainer.train()
    elif cfg.mode.action == "evaluate":
        trainer.evaluate()
    elif cfg.mode.action == "envision":
        trainer.envision(cfg.mode.envision_id)
    else:
        raise ValueError(f"Unknown mode action: {cfg.mode.action}")

if __name__ == "__main__":
    main()
