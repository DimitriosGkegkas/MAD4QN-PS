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
