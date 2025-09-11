"""
Iris Server implementation
"""
from concurrent import futures
import time
import grpc
import numpy as np

import iris_pb2_grpc as pb2_grpc
import iris_pb2 as pb2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class IrisServer(pb2_grpc.IrisServicer):
    """
    Setup the server
    """
    def __init__(self):
        self.models = {}
        self.current_model = None
     
    def GetServerResponseFit(self, request, context):
        """
        Args:
            request (FitRequest): 
                - features: flattened feature matrix (1D array).
                - labels: array of class labels.
                - rows: number of samples.
                - cols: number of features per sample.
                - model_type: type of model (e.g., "knn", "svm").
            context: gRPC context (not used here).

        Returns:
            FitResponse: containing training accuracy, training time,
                         model ID, and metadata.
        """
        start_time = time.time()
      
        # Reconstruct the 2D array from flattened features
        features = np.array(request.features).reshape(request.rows, request.cols)
        labels = np.array(request.labels)
        # Choose model based on request
        model_type = request.model_type.lower()
        if model_type == "knn":
            model = KNeighborsClassifier(n_neighbors=3)
        else:
            model = KNeighborsClassifier(n_neighbors=3)

        # Train the model
        model.fit(features, labels)

        # Calculate training accuracy
        predictions = model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        training_time = time.time() - start_time

        # Store the trained model
        model_id = f"{model_type}_{int(time.time())}"
        self.models[model_id] = model
        self.current_model = model

        return pb2.FitResponse(
            message=f"Model trained successfully! Model ID: {model_id}",
            accuracy=accuracy,
            training_time=training_time,
            model_info=f"{model_type} with {len(labels)} samples"
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
                model_info="None"
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
            model_info=f"Predictions: {predictions.tolist()}"
        )


def serve():
    """_summary_
    Initialize Server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_IrisServicer_to_server(IrisServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
