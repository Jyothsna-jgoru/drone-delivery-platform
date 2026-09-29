from packages.communication.network import CommunicationNetwork
from packages.shared.models import DroneBroadcast, MissionStatus, Priority


def message():
    return DroneBroadcast(drone_id="D1", position=(0,0,2), velocity=(1,0,0), next_waypoint=(1,0,2), planned_altitude=2, battery=.8, package_priority=Priority.NORMAL, mission_status=MissionStatus.ACTIVE)


def test_range_delay_and_delivery_are_seeded():
    network = CommunicationNetwork(5, 0, 2, seed=1)
    network.broadcast(message(), {"D1":(0,0,2),"D2":(3,0,2),"D3":(10,0,2)}, 1)
    received = network.receive("D2", 4)
    assert len(received) == 1
    assert received[0].sender == "D1"
    assert network.receive("D3", 4) == []


def test_loss_is_recorded_not_silently_discarded():
    network = CommunicationNetwork(5, 1, 0, seed=1)
    network.broadcast(message(), {"D1":(0,0,2),"D2":(3,0,2)}, 1)
    received = network.receive("D2", 1)
    assert received[0].dropped

