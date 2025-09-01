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

# x_treino, x_teste, y_treino, y_test = train_test_split(atributos, rotulos, test_size=0.2)

class IrisServer(pb2_grpc.IrisServicer):
    """
    Args:
        pb2_grpc
    """
    def __init__(self):
        self.models = {}
        self.current_model = None
    def GetServerResponseFit(self, request, context):
        """
        Args:
            request (_type_): _description_
            context (_type_): _description_

        Returns:
            mensagem_resposta: _description_
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
            model = KNeighborsClassifier(n_neighbors=3)  # default
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
    