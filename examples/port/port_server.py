"""
A federated learning server using Port.

Reference:

"How Asynchronous can Federated Learning Be?"

"""

import asyncio
import copy
import os

import torch
import torch.nn.functional as F
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm. """

    async def cosine_similarity(self, update, staleness):
        """ Compute the cosine similarity of the received updates and the difference
            between the current and a previous model according to client staleness. """
        # Loading the global model from a previous round according to staleness
        filename = f"model_{self.current_round - 2}.pth"
        model_dir = Config().params['model_dir']
        model_path = f'{model_dir}/{filename}'

        similarity = 0

        if staleness > 0 and os.path.exists(model_path):
            previous_model = copy.deepcopy(self.trainer.model)
            previous_model.load_state_dict(torch.load(model_path))

            previous = torch.zeros(0)
            for layer in previous_model.parameters():
                previous = torch.cat((previous, layer.data.view(-1)))

            current = torch.zeros(0)
            for layer in self.trainer.model.parameters():
                current = torch.cat((current, layer.data.view(-1)))

            deltas = torch.zeros(0)
            for __, delta in update.items():
                deltas = torch.cat((deltas, delta.view(-1)))

            similarity = F.cosine_similarity(current - previous, deltas, dim=0)

        return similarity

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        # Constructing the aggregation weights to be used
        aggregation_weights = []

        for i, update in enumerate(weights_received):
            report, __, staleness = updates[i]
            num_samples = report.num_samples

            similarity = await self.cosine_similarity(update, staleness)
            staleness_factor = Server.staleness_function(staleness)
            print(
                f'similarity = {similarity + 1}, staleness = {staleness}, staleness factor = {staleness_factor}'
            )
            print(
                f'for client {i}, raw weight = {num_samples / self.total_samples * (similarity + 1) * staleness_factor}'
            )

            aggregation_weights.append(num_samples / self.total_samples *
                                       (similarity + 1) * staleness_factor)

        # Normalize so that the sum of aggregation weights equals 1
        aggregation_weights = [
            i / sum(aggregation_weights) for i in aggregation_weights
        ]

        print('normalized weights = ', aggregation_weights)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __, staleness = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * aggregation_weights[i]

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

        # Save the current model for later retrieval when cosine similarity needs to be computed
        filename = f"model_{self.current_round}.pth"
        self.trainer.save_model(filename)

    @staticmethod
    def staleness_function(staleness):
        """ The staleness_function. """
        staleness_bound = 10

        if hasattr(Config().server, "staleness_bound"):
            staleness_bound = Config().server.staleness_bound

        staleness_factor = staleness_bound / (staleness + staleness_bound)

        return staleness_factor
