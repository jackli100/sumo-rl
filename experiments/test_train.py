import unittest
import os
from TrafficSimulationSuite import ShowResults

class TestTrain(unittest.TestCase):
    def test_info(self):
        # Test case 1
        show_results = ShowResults(r'D:\trg1vr\sumo-rl\autodl-tmp\outputs\2_0.7-0.7-0.7-0.7')
        show_results.drawing_from_csv()
       





if __name__ == '__main__':
    unittest.main()
