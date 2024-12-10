from .sim import SIM
from .sim import SIM
from .sim import SIM
from .sim import SIM
from .sim import SIM


MODELS = {
    "DIN": 
    
}

def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)