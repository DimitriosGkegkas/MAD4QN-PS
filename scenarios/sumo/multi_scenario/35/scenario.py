import pickle
from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    MapSpec,
    Mission,
    Route,
    Scenario,
)

current_file_path = Path(__file__).resolve()
with open(current_file_path.parent.parent / "routes.pkl", "rb") as f:
    # Load the content of the file using pickle.load()
    routes = pickle.load(f)

assert routes, "routes.pkl is empty"

# get the id based on the name of the parent folder as a number
id = int(current_file_path.parent.name)
assert routes[id], f"routes.pkl does not contain routes for scenario {id}"

gen_scenario(
    scenario=Scenario(
        map_spec=MapSpec(
            source = str(current_file_path.parent.parent / "map.net.xml"),
            shift_to_origin=False,
            lanepoint_spacing=1.0,
        ),
        traffic=None,
        ego_missions=[
            Mission(Route(begin=("edge-east-EW", 0, 10), end=(routes[id][0], 0, 60))),
            Mission(Route(begin=("edge-west-WE", 0, 10), end=(routes[id][1], 0, 60))),
            Mission(Route(begin=("edge-north-NS", 0, 10), end=(routes[id][2], 0, 60))),
            Mission(Route(begin=("edge-south-SN", 0, 10), end=(routes[id][3], 0, 60))),
        ],
    ),
    output_dir=Path(__file__).parent,
)
