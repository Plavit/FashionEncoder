from abc import ABC, abstractmethod
import random
import tensorflow as tf


class Recommender(ABC):

    @abstractmethod
    def init_dataset(self, metadata, test_ids):
        pass

    @abstractmethod
    def get_nearest(self, outfit_item_ids, category, count):
        pass


class FashionEncoderRec(Recommender):

    def __init__(self, model):
        self.model = model
        self._preprocessor = model.get_layer("preprocessor")

    def get_nearest(self, outfit_item_ids, category, count):
        pass

    def init_dataset(self, metadata, test_ids):

        pass


class RetrievalTask:

    def __init__(self, test_ids, dataset, metadata_path, recommender: Recommender):
        self._dataset = dataset
        self._recommender = recommender
        # self._metadata load metadata
        recommender.init_dataset(self._metadata, test_ids)

    def run_test(self, top_k):
        self._reset_metrics()
        random.seed(123)

        for outfit in self._dataset:
            outfit_item_ids = [item.item_id for item in outfit.items]
            target = random.choice(outfit_item_ids)
            outfit_item_ids.remove(target)

            top_k = self._recommender.get_nearest(outfit_item_ids, self._metadata[target].semantic_category, top_k)

            self._update_metrics(target, top_k)

    def _reset_metrics(self):
        pass

    def _update_metrics(self, target, top_k):
        pass