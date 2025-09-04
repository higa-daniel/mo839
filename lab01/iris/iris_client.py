"""
Iris Client implementation
"""
import grpc
import iris_pb2_grpc as pb2_grpc
import iris_pb2 as pb2
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class IrisClient(object):
    """
    Setup the client connection and prepare the data.
    """
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub = pb2_grpc.IrisStub(self.channel)
        self.model_type = None

        iris = load_iris()
        atributos = iris.data
        rotulos = iris.target
        self.x_treino, self.x_teste, self.y_treino, self.y_teste = train_test_split(atributos, rotulos, test_size=0.2)

    def get_fit_response(self):
        """
        Args:
            None
        Returns:
            pb2.FitResponse: A protocol buffer message containing the training results.
                             This includes a message, training accuracy, and training time.
        """
        self.model_type = "knn"

        # Use pre-loaded data
        flattened_features = self.x_treino.flatten().tolist()
        request = pb2.FitRequest(
            features=flattened_features,
            labels=self.y_treino.tolist(),
            rows=self.x_treino.shape[0],
            cols=self.x_treino.shape[1],
            model_type=self.model_type
        )
        print(f"Treinando modelo {self.model_type.upper()} com {len(self.y_treino)} amostras...")
        response = self.stub.GetServerResponseFit(request)
        print("Treinamento concluído!")
        print(f"Mensagem: {response.message}")
        print(f"Acurácia de Treino: {response.accuracy:.4f}")
        print(f"Tempo de Treinamento: {response.training_time:.4f} segundos")
        print(f"Informações do Modelo: {response.model_info}")
        print("-" * 50)

    def get_predict_response(self):
        """
        Args:
            None
        Returns:
            pb2.PredictResponse: A protocol buffer message with prediction results, 
            including accuracy and predictions.
        """
        flattened_features = self.x_teste.flatten().tolist()
        request = pb2.PredictRequest(
            features=flattened_features,
            labels=self.y_teste.tolist(),
            rows=self.x_teste.shape[0],
            cols=self.x_teste.shape[1],
            model_type=self.model_type
        )
        print(f"Testando modelo {self.model_type.upper()} com {len(self.y_teste)} amostras...")
        response = self.stub.GetServerResponsePredict(request)
        print("Teste concluído!")
        print(f"Mensagem: {response.message}")
        print(f"Acurácia de Teste: {response.accuracy:.4f}")
        print(f"Tempo de teste: {response.prediction_time:.4f} segundos")
        print(f"Informações do Modelo: {response.model_info}")
        print("-" * 50)
        return response

if __name__ == '__main__':
    client = IrisClient()
    client.get_fit_response()
    client.get_predict_response()
    