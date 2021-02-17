from models.ganite import GANITE
from models.cmgp import CMGP


class Model:

    def __init__(self, model: str, *args, **kwargs):
        if model == 'GANITE':
            self.model = GANITE(*args, **kwargs)
        elif model == 'CMGP':
            self.model = CMGP(*args, **kwargs)
        else:
            raise ValueError(
                f"Undefined model specified : {model}. "
            ) 
    