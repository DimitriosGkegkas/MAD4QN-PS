from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import Scenario, Traffic, Flow, Route, RandomRoute, TrafficActor, Distribution, MapSpec
import pickle


traffic_actor = TrafficActor(
    name="car",
)

current_file_path = Path(__file__).resolve()

with open( str(current_file_path.parent.parent.parent / "routes.pkl"), "rb") as f:
    # Load the content of the file using pickle.load()
    routes = pickle.load(f)
assert routes, "routes.pkl is empty"

# get the id based on the name of the parent folder as a number
id = int(current_file_path.parent.name)
assert routes[id], f"routes.pkl does not contain routes for scenario {id}"

traffic = Traffic(flows=[
    Flow(
        route=Route(begin=("edge-east-EW", 0, 10), end=(routes[id][0], 0, 60)),
            begin=0,
            end=10 * 60 * 60,  # Flow lasts for 10 hours.
            rate=1,  # One vehicle enters per second.
            actors={traffic_actor: 1},
    ),
    Flow(
        route=Route(begin=("edge-west-WE", 0, 10), end=(routes[id][1], 0, 60)),
            begin=0,
            end=10 * 60 * 60,  # Flow lasts for 10 hours.
            rate=1,  # One vehicle enters per second.
            actors={traffic_actor: 1},
    ),
    Flow(
        route=Route(begin=("edge-north-NS", 0, 10), end=(routes[id][2], 0, 60)),
            begin=0,
            end=10 * 60 * 60,  # Flow lasts for 10 hours.
            rate=1,  # One vehicle enters per second.
            actors={traffic_actor: 1},
    ),
    Flow(
        route=Route(begin=("edge-south-SN", 0, 10), end=(routes[id][3], 0, 60)),
            begin=0,
            end=10 * 60 * 60,  # Flow lasts for 10 hours.
            rate=1,  # One vehicle enters per second.
            actors={traffic_actor: 1},
    ),
])

# Generate the scenario

gen_scenario(
    scenario=Scenario(
        traffic={"basic": traffic},
        map_spec=MapSpec(
            source = str(current_file_path.parent / "map.net.xml"),
        ),
        ego_missions=[]
    ), 
    output_dir=Path(__file__).parent,
)
