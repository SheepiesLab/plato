from turtle import update
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

    async def federated_averaging(self, updates, alpha1=None, alpha2=None):

        if self.current_round <= 1:  # first fl round using plain aggregation
            update = await super().federated_averaging(updates)
        else: # adaptive aggregation with the help of DRL agent
            



        return update
