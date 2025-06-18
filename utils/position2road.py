### 
# E = [0, 1]
# W = [0, 0]
# N = [1, 0]
# S = [1, 1]

# so e.g. EW = [0, 1, 0, 0] (x1, y1, x2, y2)

# """


roads_to_direction = {
    'EW': [0, 1, 0, 0],
    'ES': [0, 1, 1, 1],
    'EN': [0, 1, 1, 0],
    
    'WE': [0, 0, 0, 1],
    'WN': [0, 0, 1, 0],
    'WS': [0, 0, 1, 1],
    
    'SN': [1, 1, 1, 0],
    'SW': [1, 1, 0, 0],
    'SE': [1, 1, 0, 1],
    
    'NS': [1, 0, 1, 1],
    'NE': [1, 0, 0, 1],
    'NW': [1, 0, 0, 0], 
}

road_to_communication = {
    'E': ['S', 'W', 'N'],
    'W': ['N', 'E', 'S'],
    'S': ['W', 'N', 'E'],
    'N': ['E', 'S', 'W'],
}


def position_to_road(position):
    if position[0] > 50:
        return 'E'
    elif position[0] < 30:
        return 'W'
    elif position[1] < 30:
        return 'S'
    else:
        return 'N'

# Define conflict rules based on common intersection conflicts
conflict_rules = {
    "WE": { "WS", "WN", "NS", "NE", "ES", "SN", "SE", "SW" },
    "EW": { "ES", "EN", "NS", "NW", "NE", "WN", "SN", "SW" },
    "SN": { "SW", "SE", "EW", "EN", "ES", "NE", "WE", "WN" },
    "NS": { "NE", "NW", "SW", "WE", "EW", "WS", "ES", "WN" },
    
    "EN": { "EW", "ES", "WN", "SN" },
    "NW": { "NS", "NE", "SW", "EW" },
    "SE": { "SN", "SW", "WE", "NE" },
    "WS": { "WN", "WE", "ES", "NS" },
    
    "ES": { "EW", "EN", "WS", "NS", "WE", "SN", "WN", "NE", "SW" },
    "WN": { "WE", "WS", "EN", "SN", "EW", "NS", "ES", "NE", "SW" },
    "SW": { "SN", "SE", "NW", "EW", "NS", "WE", "NE", "ES", "WN" },
    "NE": { "NS", "NW", "WE", "NE",  "SN", "EW", "WN", "SW", "ES" },
}


def has_conflict(target_route: str, route_list: list) -> list:
    """
    Given a target vehicle route (e.g., 'EW') and a list of other vehicle routes,
    returns a list of booleans indicating which routes have a conflict with the target.
    """
    # Check for conflicts
    return [route in conflict_rules.get(target_route, set()) for route in route_list]

def has_conflict_v2v(target_route: str, check_route: str) -> bool:
    """
    Given a target vehicle route (e.g., 'EW') and another vehicle route,
    returns a boolean indicating whether the two routes have a conflict.
    """
    return check_route in conflict_rules.get(target_route, set())
