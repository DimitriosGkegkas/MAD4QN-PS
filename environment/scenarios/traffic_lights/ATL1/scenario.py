from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import Scenario, MapSpec


current_file_path = Path(__file__).resolve()

# Define the map and traffic
map_spec = MapSpec(source= str(current_file_path.parent / "map.net.xml"), shift_to_origin = True)
assert map_spec, "Map not found"

# Generate the scenario

gen_scenario(
    scenario=Scenario(
        map_spec=map_spec,  
    ), 
    output_dir= str(current_file_path.parent),
)
