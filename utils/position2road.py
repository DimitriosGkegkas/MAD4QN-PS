roads2t_i = {'EW': 'straight', 'ES': 'left', 'EN': 'right',
             'WE': 'straight', 'WN': 'left', 'WS': 'right',
             'SN': 'straight', 'SW': 'left', 'SE': 'right',
             'NS': 'straight', 'NE': 'left', 'NW': 'right'}


def position2road(position):
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
