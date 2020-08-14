import unittest

from revolve.robot.robot import Robot
from simulation.simulator.simulation_connector import SimulatorState
from evosphere.TestEcosphere import TestEcosphere
from simulation_test.simulator.test_connector_adapter import TestConnectorAdapter


class TestSimulatorConnector(unittest.TestCase):

    def test_connector_port(self):
        connector_1 = TestConnectorAdapter(TestEcosphere())
        connector_2 = TestConnectorAdapter(TestEcosphere())

        self.assertNotEqual(connector_1.port, connector_2.port)

    def test_connector_ready(self):
        connector = TestConnectorAdapter(TestEcosphere())

        self.assertEqual(connector.state, SimulatorState.READY)

        connector.stop()

        self.assertEqual(connector.state, SimulatorState.STOPPED)

        connector.restart()

        self.assertEqual(connector.state, SimulatorState.READY)

        connector.restart()

        self.assertEqual(connector.state, SimulatorState.READY)

    def test_connector_robot(self):
        connector = TestConnectorAdapter(TestEcosphere())

        robot = Robot()
        connector.add_robot(robot)

        connector.remove_robot(robot)

        self.assertTrue(True)

    def test_connector_robot_changed(self):
        connector = TestConnectorAdapter(TestEcosphere())

        robot = Robot()
        connector.add_robot(robot)

        self.assertRaises(Exception, connector.remove_robot, Robot())
