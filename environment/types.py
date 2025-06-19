from typing import Dict, List, Tuple

import numpy as np

ImageObservationType = np.ndarray
MessageObservationType = np.ndarray
RawMessageObservationType = List[np.ndarray]
DirectionObservationType = np.ndarray


ObservationType = Tuple[ImageObservationType, DirectionObservationType, MessageObservationType, RawMessageObservationType]