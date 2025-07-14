import time
import gymnasium as gym
import numpy as np
from smarts.core.utils.core_math import line_intersect
from utils import position_to_road, roads_to_direction, road_to_communication


class InfoWrapper(gym.Wrapper):
    def __init__(self, env, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super(InfoWrapper, self).__init__(env)
        self.agent_names = agent_names
        self.env = env
        
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        termination, truncation, reward = self._create_dummy_reset_output()

        return (
            observation,
            termination,
            truncation,
            reward,
            info,
        )
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self.add_time_separation(info)
        return obs, reward, terminated, truncated, info 
    
    def _create_dummy_reset_output(self):
        return (
            {agent: False for agent in self.agent_names},   # termination
            {agent: False for agent in self.agent_names},   # truncation
            {agent: 0 for agent in self.agent_names},       # reward
        )
        
    def add_time_separation(self, info):
        time_separation_dict = compute_all_time_separations(info, window=1.0)
        for agent_name, tsep in time_separation_dict.items():
            if agent_name in info:
                info[agent_name]['time_separation'] = np.exp(-0.5 * tsep)
        return info
        
    





def find_path_intersection(path1, path2, tol=0.5):
    """
    Given two waypoint paths, find the first intersection point.

    Args:
        path1, path2: Lists of Waypoint objects or dicts with .pos or ['pos'] attributes.
        tol: Slack tolerance (still used to expand usefulness).

    Returns:
        [x, y] intersection point or None.
    """
    def get_pos(wp):
        return np.array(wp.pos[:2]) if hasattr(wp, 'pos') else np.array(wp['pos'][:2])

    for i in range(len(path1) - 1):
        p1 = get_pos(path1[i])
        p2 = get_pos(path1[i + 1])
        for j in range(len(path2) - 1):
            q1 = get_pos(path2[j])
            q2 = get_pos(path2[j + 1])
            intersection = line_intersect(p1, p2, q1, q2)
            if intersection is not None:
                # Optional slack tolerance
                d1 = np.linalg.norm(p2 - p1)
                d2 = np.linalg.norm(q2 - q1)
                if np.linalg.norm(intersection - p1) <= d1 + tol and np.linalg.norm(intersection - q1) <= d2 + tol:
                    return intersection.tolist()
            else:
                # Check for coincident segments
                if are_colinear(p1, p2, q1, q2):
                    # Return first shared point (either q1 or q2 if on segment p1-p2)
                    if point_on_segment(q1, p1, p2, tol):
                        return q1.tolist()
                    elif point_on_segment(q2, p1, p2, tol):
                        return q2.tolist()
    return None

def are_colinear(a1, a2, b1, b2, eps=1e-8):
    # Check if two segments are colinear using cross product
    v1 = a2 - a1
    v2 = b1 - a1
    v3 = b2 - a1
    return np.abs(np.cross(v1, v2)) < eps and np.abs(np.cross(v1, v3)) < eps

def point_on_segment(p, a, b, tol=1e-8):
    # Check if point p lies on segment ab
    ap = p - a
    ab = b - a
    proj = np.dot(ap, ab) / np.dot(ab, ab)
    return 0 - tol <= proj <= 1 + tol and np.linalg.norm(np.cross(ap, ab)) / np.linalg.norm(ab) < tol

def distance_along_path_to_point(position, waypoints, target_point, tol=0.5):
    """
    Computes distance along a polyline from current position to a target point (e.g., intersection).
    Stops once target point is within `tol` of any segment.
    
    Args:
        position: Current [x, y]
        waypoints: List of Waypoints or dicts with 'pos'
        target_point: [x, y]
        tol: Distance tolerance to consider a segment contains the point

    Returns:
        Total path distance from current position to target_point, or None if not on path.
    """
    def get_pos(wp):
        return np.array(wp.pos[:2]) if hasattr(wp, 'pos') else np.array(wp['pos'][:2])
    
    pos = np.array(position[:2])
    target = np.array(target_point[:2])
    
    # Find the closest segment to current position to start from
    start_index = 0
    for i in range(len(waypoints) - 1):
        seg_start = get_pos(waypoints[i])
        seg_end = get_pos(waypoints[i + 1])
        if np.linalg.norm(pos - seg_start) + np.linalg.norm(pos - seg_end) - np.linalg.norm(seg_end - seg_start) < tol:
            start_index = i
            break

    dist = 0.0
    reached = False
    for i in range(start_index, len(waypoints) - 1):
        a = get_pos(waypoints[i])
        b = get_pos(waypoints[i + 1])
        segment_length = np.linalg.norm(b - a)

        # Check if the intersection point lies within this segment
        if (
            np.linalg.norm(a - target) + np.linalg.norm(b - target)
            - segment_length
            < tol
        ):
            dist += np.linalg.norm(target - a)
            reached = True
            break
        else:
            dist += segment_length

    return dist if reached else None

def estimated_time_to_point(position, velocity, waypoints, intersection, tol=0.5):
    distance = distance_along_path_to_point(position, waypoints, intersection, tol)
    if distance is None:
        return float("inf")  # Not reachable on path
    speed = np.linalg.norm(velocity[:2])
    if speed < 1e-3:  # Avoid division by zero
        return float("inf")
    return distance / speed

def time_separation(agent1, agent2, window=6.0, tol=0.5):
    # print("=" * 60)
    # print(f"[DEBUG] Comparing agents: {agent1['name']} <-> {agent2['name']}")

    intersection = find_path_intersection(agent1['path'], agent2['path'], tol=tol)
    if intersection is None:
        # print("[DEBUG] No intersection found between paths.")
        return np.inf, np.inf

    # print(f"[DEBUG] Intersection point: {intersection}")

    # Time estimates
    t1 = estimated_time_to_point(agent1['position'], agent1['velocity'], agent1['path'], intersection, tol)
    t2 = estimated_time_to_point(agent2['position'], agent2['velocity'], agent2['path'], intersection, tol)

    # print(f"[DEBUG] {agent1['name']} → time to intersection: {t1:.3f} sec")
    # print(f"[DEBUG] {agent2['name']} → time to intersection: {t2:.3f} sec")

    if min(t1, t2) > window:
        # print(f"[DEBUG] Both agents are beyond the window ({window}s). Ignoring.")
        return np.inf, np.inf

    if t2 > t1:
        # print(f"[DEBUG] {agent1['name']} arrives first. {agent2['name']} gets separation: {t2 - t1:.3f}s")
        return np.inf, t2 - t1
    else:
        # print(f"[DEBUG] {agent2['name']} arrives first. {agent1['name']} gets separation: {t1 - t2:.3f}s")
        return t1 - t2, np.inf

    
    
def compute_all_time_separations(info: dict, window=6.0, tol=1) -> dict:
    from collections import defaultdict
    time_separations = defaultdict(lambda: np.inf)

    # Prepare agent snapshots
    agents = {}
    for name, data in info.items():
        if 'env_obs' not in data:
            continue
        state = data['env_obs'].ego_vehicle_state
        agents[name] = {
            'position': np.array(state.position[:2]),
            'velocity': np.array(state.linear_velocity[:2]),
            'path': data['env_obs'].waypoint_paths[0],  # Assuming path 0
            'name': name,
        }

    agent_names = list(agents.keys())

    # Pairwise comparisons
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a1 = agent_names[i]
            a2 = agent_names[j]
            sep1, sep2 = time_separation(
                agents[a1], agents[a2], window=window, tol=tol
            )
            time_separations[a1] = min(time_separations[a1], sep1)
            time_separations[a2] = min(time_separations[a2], sep2)

    return dict(time_separations)
