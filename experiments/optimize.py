from TrafficSimulationSuite import Train

if __name__ == "__main__":

    proportion_of_saturations = [0.75, 0.75, 0.75, 0.75]
    net_path = r"D:\trg1vr\sumo-rl\sumo_rl\nets\2way-single-intersection\2-stage.xml"
    total_timesteps = 500000
    num_of_episodes = 5

    # Initialize the trainer
    trainer = Train(
        net_file=net_path,
        total_timesteps=total_timesteps,
        num_of_episodes=num_of_episodes,
    )

    # Train the model
    trainer.optimize_hyperparameters()
    
