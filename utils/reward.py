from smarts.core.utils.core_math import signed_dist_to_line, radians_to_vec
import math
import numpy as np

def dist_to(a, b) -> float:
    return np.linalg.norm(np.array(a) - np.array(b))

def _get_vehicle_lookahead_point(position, heading, target_point) -> list:
    look_ahead_dist = dist_to(target_point[:2], position[:2])
    return [
        position[0] - look_ahead_dist * math.sin(heading),
        position[1] + look_ahead_dist * math.cos(heading),
    ]

def _get_lookahead_waypoint(obs: dict, look_ahead_wp_num: int):
    wp_positions = obs["waypoint_paths"]["position"][0]
    wp_headings = obs["waypoint_paths"]["heading"][0]
    look_ahead_wp = wp_positions[look_ahead_wp_num]
    look_ahead_wp_head = wp_headings[look_ahead_wp_num]
    return look_ahead_wp, look_ahead_wp_head

def get_lateral_error(obs: dict, look_ahead_wp_num: int = 4) -> float:
    position = obs["ego_vehicle_state"]["position"]
    heading = obs["ego_vehicle_state"]["heading"]
    look_ahead_wp, look_ahead_wp_head = _get_lookahead_waypoint(obs, look_ahead_wp_num)

    if np.allclose(look_ahead_wp, 0.0):
        return 0.0

    vehicle_look_ahead_pt = _get_vehicle_lookahead_point(position, heading, look_ahead_wp)

    lat_error = signed_dist_to_line(
        vehicle_look_ahead_pt, look_ahead_wp[:2], radians_to_vec(look_ahead_wp_head)
    )
    return np.clip(abs(lat_error), -0.1, 10)
