from sklearn.cluster import KMeans
import numpy as np
import supervision as sv

import torch
import umap
from transformers import AutoProcessor, SiglipVisionModel

from typing import Iterable, Generator, List, TypeVar

V = TypeVar("V")

MODEL_PATH = 'google/siglip2-base-patch16-224'

def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch

class TeamAssigner:
    def __init__(self, batch_size: int = 32):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(MODEL_PATH, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
        self.reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        self.player_team = {}

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)
    
    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.kmeans.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.kmeans.predict(projections)

    def get_player_team(self, frame: np.ndarray, bbox: list, player_id: int) -> int:
        """Gets the player team from the frame using KMeans clustering.

        Args:
            frame (np.ndarray): Frame to get the player team from.
            bbox (list): Bounding box of the player.
            player_id (int): The ID of the player.

        Returns:
            int: The team ID of the player.
        """
        if player_id in self.player_team:
            return self.player_team[player_id]

        crop = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
        team = self.predict(crop)[0]

        self.player_team[player_id] = team
        return team
