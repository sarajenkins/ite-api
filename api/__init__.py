from .models.ganite import GANITE
from .models.cmgp import CMGP


class Model:
    def __init__(self, model: str, *args, **kwargs):
        if model == 'GANITE':
            self.model = GANITE(*args, **kwargs)
        elif model == 'CMGP':
            self.model = CMGP(*args, **kwargs)
        else:
            raise ValueError(f"Undefined model specified : {model}. ")

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def test(self, *args, **kwargs):
        return self.model.test(*args, **kwargs)
