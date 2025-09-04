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
    Setup the client connection and prepare the data
    """
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub        = pb2_grpc.IrisStub(self.channel) 

    def get_fit_response(self):
        """
        Args:
            None
        Returns:
            pb2.FitResponse: A protocol buffer message containing the training results.
                             This includes a message, training accuracy, and training time.
        # """
        # Define model
        model_type="knn"
        # Load and prepare data
        iris = load_iris()
        atributos = iris.data
        rotulos = iris.target
        # Split the data
        x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, rotulos, test_size=0.2)
        # Flatten the 2D array to send to server
        flattened_features = x_treino.flatten().tolist()
        request = pb2.FitRequest(
            features=flattened_features,
            labels=y_treino.tolist(),
            rows=x_treino.shape[0],
            cols=x_treino.shape[1],
            model_type=model_type
        )
        print(f"Treinando modelo {model_type.upper()} com {len(y_treino)} amostras...")
        response = self.stub.GetServerResponseFit(request)
        print(f"Treinamento concluído!")
        print(f"Mensagem: {response.message}")
        print(f"Acurácia de Treino: {response.accuracy:.4f}")
        print(f"Tempo de Treinamento: {response.training_time:.4f} segundos")
        print(f"Informações do Modelo: {response.model_info}")
        print("-" * 50)
        # Store the test data for later prediction
        self.x_teste = x_teste
        self.y_teste = y_teste
        return self.stub.GetServerResponseFit(request)

    def get_predict_response(self):
        """
        Args:
            message (_type_): _description_
        Returns:
            _type_: _description_
        # """
        # Define model
        model_type="knn"
        # Load and prepare data
        iris = load_iris()
        atributos = iris.data
        rotulos = iris.target
        # Split the data
        x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, rotulos, test_size=0.2)
        # Flatten the 2D array to send to server
        flattened_features = x_teste.flatten().tolist()
        request = pb2.FitRequest(
            features=flattened_features,
            labels=y_teste.tolist(),
            rows=x_teste.shape[0],
            cols=x_teste.shape[1],
            model_type=model_type
        )
        print(f"Testabdi modelo {model_type.upper()} com {len(y_teste)} amostras...")
        response = self.stub.GetServerResponseFit(request)
        print(f"Teste concluído!")
        print(f"Mensagem: {response.message}")
        print(f"Acurácia de Teste: {response.accuracy:.4f}")
        print(f"Tempo de teste: {response.training_time:.4f} segundos")
        print(f"Informações do Modelo: {response.model_info}")
        print("-" * 50)
        # Store the test data for later prediction
        self.x_teste = x_teste
        self.y_teste = y_teste
        return self.stub.GetServerResponseFit(request)

if __name__ == '__main__':
    client   = IrisClient()
    client.get_fit_response()
    client.get_predict_response()
    