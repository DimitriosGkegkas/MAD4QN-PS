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

gen_scenario(
    scenario=Scenario(
        map_spec=MapSpec(
            source = str(current_file_path.parent.parent / "map.net.xml"),
            shift_to_origin = True
        ),
        traffic=None,
        ego_missions=[
            Mission(Route(begin=("edge-south-SN", 0, 10), end=("edge-west-EW", 0, 60))),
        ],
    ),
    output_dir=Path(__file__).parent,
)
