"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from plato.algorithms import registry as algorithms_registry
from plato.clients import base
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    update_response: bool


class Client(base.Client):
    """A basic federated learning client who sends simple weight updates."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__()
        self.model = model
        self.datasource = datasource
        self.algorithm = algorithm
        self.trainer = trainer
        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.sampler = None
        self.test_set_sampler = None  # Sampler for the test set

        self.report = None

        self.varied_partition = 1
        self.varied_epochs = 1

    def __repr__(self):
        return 'Client #{}.'.format(self.client_id)

    def configure(self) -> None:
        """Prepare this client for training."""
        super().configure()

        if self.trainer is None:
            self.trainer = trainers_registry.get(self.model)
        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)
        self.algorithm.set_client_id(self.client_id)

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer)

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        # Reset workload capacity at the initial step (for the new episode)
        if server_response['current_round'] == 1:
            # Clients claim different number of data samples in each new episode
            if hasattr(Config().data, 'varied') and Config().data.varied:
                # self.varied_partition = np.random.choice([0.5, 0.8, 1.0, 1.25, 2.0])
                self.varied_partition = np.random.choice([0.1, 0.4, 0.7, 1.0])

            # Clients claim different number of training epochs in each new episode
            if hasattr(Config().trainer, 'varied') and Config().trainer.varied:
                # self.varied_epochs = np.random.choice([0.5, 0.8, 1.0, 1.25, 2.0])
                # self.varied_epochs = np.random.choice([0.1, 0.3, 0.5, 0.7, 1.0])
                self.varied_epochs = np.random.choice([0.1, 0.4, 0.7, 1.0])

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        logging.info("[Client #%d] Loading its data source...", self.client_id)

        if self.datasource is None or (hasattr(Config().data, 'reload_data')
                                       and Config().data.reload_data):
            self.datasource = datasources_registry.get(
                client_id=self.client_id)

        self.data_loaded = True

        logging.info("[Client #%d] Dataset size: %s", self.client_id,
                     self.datasource.num_train_examples())

        # Setting up the data sampler
        self.sampler = samplers_registry.get(self.datasource, self.client_id, self.varied_partition)

        if hasattr(Config().trainer, 'use_mindspore'):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, 'test_set_sampler'):
                # Set the sampler for test set
                self.test_set_sampler = samplers_registry.get(self.datasource,
                                                              self.client_id,
                                                              testing=True)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""
        self.algorithm.load_weights(server_payload)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler, self.varied_epochs)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset, self.test_set_sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                self.client_id, 100 * accuracy))
        else:
            accuracy = 0

        if hasattr(Config().clients,
                   'sleep_simulation') and Config().clients.sleep_simulation:
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time

            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 (avg_training_time + sleep_seconds) *
                                 Config().trainer.epochs, False)
        else:
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 training_time, False)

        return self.report, weights

    async def obtain_model_update(self, wall_time):
        """Retrieving a model update corresponding to a particular wall clock time."""
        model = self.trainer.obtain_model_update(wall_time)
        weights = self.algorithm.extract_weights(model)
        self.report.update_response = True

        return self.report, weights
