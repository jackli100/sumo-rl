import unittest
import os
from TrafficSimulationSuite import Train, TrafficMatrix, generate_result_folder, ShowResults

class TestTrain(unittest.TestCase):
    def test_info(self):
        # Test case 1
        train_instance = Train(output_folder="output", net_file=r"sumo_rl/nets/2way-single-intersection/17-stage.xml", route_file="route.xml", fix_ts=True)
        self.assertEqual(train_instance.net_info, "17_fixed")
        # Test case 2
        traffic_matrix = TrafficMatrix("output", [0.75, 0.75, 0.75, 0.75])
        self.assertEqual(traffic_matrix.get_traffic_string(), "0.75-0.75-0.75-0.75")
        # get the above two strings and concatenate them
        folder_name = train_instance.net_info + "_" + traffic_matrix.get_traffic_string()
        self.assertEqual(folder_name, "17_fixed_0.75-0.75-0.75-0.75")
        # Test case 3
        results_folder = generate_result_folder(train_instance.net_info, traffic_matrix.get_traffic_string())
        self.assertEqual(results_folder, "autodl-tmp\\outputs\\17_fixed_0.75-0.75-0.75-0.75")
        # Test case 4
        result_folder = r"D:\trg1vr\sumo-rl\16-49-54-1684"





if __name__ == '__main__':
    unittest.main()
