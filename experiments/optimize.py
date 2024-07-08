from TrafficSimulationSuite import generate_result_folder, TrafficMatrix, Train

if __name__ == "__main__":

    proportion_of_saturations = [0.75, 0.75, 0.75, 0.75]
    net_path = r"D:\trg1vr\sumo-rl\sumo_rl\nets\2way-single-intersection\2-stage.xml"
    total_timesteps = 500000
    num_of_episodes = 5
    results_folder = generate_result_folder()

    # Generate the traffic matrix
    matrix = TrafficMatrix(results_folder, proportion_of_saturations)
    matrix.create_xml()
    route_path = matrix.output_file

    # Initialize the trainer
    trainer = Train(
        output_folder=results_folder,
        net_file=net_path,
        route_file=route_path,
        total_timesteps=total_timesteps,
        num_of_episodes=num_of_episodes,
    )

    # Train the model
    trainer.optimize_hyperparameters()
    
