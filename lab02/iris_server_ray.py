"""
Iris Server implementation with Byzantine detection
"""

from concurrent import futures
import time
from collections import defaultdict
import grpc
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
import ray

import iris_pb2_grpc as pb2_grpc
import iris_pb2 as pb2

@ray.remote
class ByzantineDetector:
    """
    Ray actor for Byzantine detection
    """

    def __init__(self):
        self.detection_history = defaultdict(list)

    def detect_byzantine(self, features, labels, client_data):
        """
        Detect Byzantine clients using 3 methods

        Args:
            features: array of float 
            labels: array of int
            client_data

        Returns:
            scores: return value used to detect a byzantine client
        """

        scores = {}

        # 1st method: feature distribution anomaly
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)

        # Compare with global statistics
        global_feature_means = np.mean(
            np.vstack([d["features"] for d in client_data.values()]), axis=0
        )
        global_feature_stds = np.std(
            np.vstack([d["features"] for d in client_data.values()]), axis=0
        )

        feature_distance = np.sqrt(np.sum((feature_means - global_feature_means) ** 2))
        std_distance = np.sqrt(np.sum((feature_stds - global_feature_stds) ** 2))

        # Method 2: Label distribution anomaly
        label_dist = np.bincount(labels) / len(labels)
        global_labels = np.concatenate([d["labels"] for d in client_data.values()])
        global_label_dist = np.bincount(global_labels) / len(global_labels)

        label_distance = np.sqrt(np.sum((label_dist - global_label_dist) ** 2))

        # Method 3: Isolation Forest
        if len(client_data) > 1:
            all_features = np.vstack([d["features"] for d in client_data.values()])
            iso_forest = IsolationForest(contamination=0.1)
            iso_scores = iso_forest.fit_predict(all_features)
            iso_score = np.mean(
                iso_scores[: len(features)] == -1
            )  # Percentage of anomalies
        else:
            iso_score = 0

        # Combined score
        total_score = (
            feature_distance * 0.4
            + std_distance * 0.3
            + label_distance * 0.2
            + iso_score * 0.1
        )
        is_byzantine = total_score > 0.8  # Threshold

        scores = {
            "feature_distance": feature_distance,
            "std_distance": std_distance,
            "label_distance": label_distance,
            "iso_score": iso_score,
            "total_score": total_score,
            "is_byzantine": is_byzantine,
        }

        return scores


class IrisServer(pb2_grpc.IrisServicer):
    """
    Server with Byzantine detection
    """

    def __init__(self):
        self.models = {}
        self.current_model = None # We used KNN model
        self.client_data = {}  # Store data from all clients in dict {client_id: {"features": X, "labels": y, ...}}.
        self.byzantine_clients = set() # Set of reject ids
        self.detector = ByzantineDetector.remote() # ray actor to execute the byzantine detection logic

    def GetServerResponseFit(self, request, context):
        """
        Train model with Byzantine detection
        """
        start_time = time.time()

        # Reconstruct the 2D array
        features = np.array(request.features).reshape(request.rows, request.cols)
        labels = np.array(request.labels)
        client_id = request.client_id

        # Store client data for Byzantine detection
        self.client_data[client_id] = {
            "features": features,
            "labels": labels,
            "model_type": request.model_type,
        }

        # Detect Byzantine clients
        if len(self.client_data) > 1:
            detection_result = ray.get(
                self.detector.detect_byzantine.remote(
                    features, labels, self.client_data
                )
            )

            if detection_result["is_byzantine"]:
                self.byzantine_clients.add(client_id)
                print(
                    f"⚠️  Client {client_id} detected as Byzantine! Score: {detection_result['total_score']:.3f}"
                )
                return pb2.FitResponse(
                    message=f"Client {client_id} detected as Byzantine. Data rejected.",
                    accuracy=0.0,
                    training_time=0.0,
                    model_info="Byzantine client detected",
                )

        # Train model only with non-Byzantine data
        non_byzantine_data = {
            cid: data
            for cid, data in self.client_data.items()
            if cid not in self.byzantine_clients
        }

        if non_byzantine_data:
            all_features = np.vstack(
                [data["features"] for data in non_byzantine_data.values()]
            )
            all_labels = np.concatenate(
                [data["labels"] for data in non_byzantine_data.values()]
            )

            model_type = request.model_type.lower()
            if model_type == "knn":
                model = KNeighborsClassifier(n_neighbors=3)
            else:
                model = KNeighborsClassifier(n_neighbors=3)

            model.fit(all_features, all_labels)
            predictions = model.predict(all_features)
            accuracy = accuracy_score(all_labels, predictions)

            self.current_model = model
            training_time = time.time() - start_time

            return pb2.FitResponse(
                message=f"Model trained successfully with {len(non_byzantine_data)} clients!",
                accuracy=accuracy,
                training_time=training_time,
                model_info=f"{model_type} trained on {len(all_labels)} samples",
            )
        else:
            return pb2.FitResponse(
                message="No valid data available for training.",
                accuracy=0.0,
                training_time=0.0,
                model_info="No non-Byzantine clients",
            )

    def GetServerResponsePredict(self, request, context):
        """
        Uses the last trained model to predict labels for new data.

        Args:
            request (PredictRequest):
                - features: flattened feature matrix (1D array).
                - labels (optional): true labels for accuracy calculation.
                - rows: number of samples.
                - cols: number of features per sample.
                - model_type: ignored (uses the last trained model).
            context: gRPC context (not used here).

        Returns:
            PredictResponse: containing predictions (as metadata),
                             accuracy (if labels provided),
                             and prediction time.
        """
        if self.current_model is None:
            return pb2.PredictResponse(
                message="No model trained yet. Please train a model first.",
                accuracy=0.0,
                prediction_time=0.0,
                model_info="None",
            )
        start_time = time.time()

        # Reconstruct feature matrix
        features = np.array(request.features).reshape(request.rows, request.cols)

        # Make predictions
        predictions = self.current_model.predict(features)

        # Compute accuracy if labels are provided
        if request.labels:
            labels = np.array(request.labels)
            accuracy = accuracy_score(labels, predictions)
        else:
            accuracy = 0.0  # sem labels → não dá pra medir

        prediction_time = time.time() - start_time

        return pb2.PredictResponse(
            message="Prediction completed successfully!",
            accuracy=accuracy,
            prediction_time=prediction_time,
            model_info=f"Predictions: {predictions.tolist()}",
        )


def serve():
    """
    Initialize Server
    """
    ray.init()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_IrisServicer_to_server(IrisServer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started at 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
