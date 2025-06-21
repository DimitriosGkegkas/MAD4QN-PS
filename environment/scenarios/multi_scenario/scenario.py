import pickle
from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    MapSpec,
    Scenario,
)

current_file_path = Path(__file__).resolve()

gen_scenario(
    scenario=Scenario(
        map_spec=MapSpec(
            source = str(current_file_path.parent / "map.net.xml"),
            shift_to_origin = True
        ),
        traffic=None,
    ),
    output_dir=Path(__file__).parent,
)
