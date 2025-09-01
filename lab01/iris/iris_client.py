"""_summary_
Iris Client implementation
"""
import time
import grpc
import iris_pb2_grpc as pb2_grpc
import iris_pb2 as pb2
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class IrisClient(object):
    """_summary_
    Setup the client connection
    """
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub        = pb2_grpc.IrisStub(self.channel)

    def get_fit_response(self):
        """_summary_
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
        x_treino, x_teste, y_treino, y_test = train_test_split(atributos, rotulos, test_size=0.2)
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
        self.y_test = y_test
        return self.stub.GetServerResponseFit(request)

if __name__ == '__main__':
    client   = IrisClient()

    # while True:
        # Constrói mensagem
        # MENSAGEM = 'Fit Request'
        # tempo = time.time()
        # print(f'Cliente -> {MENSAGEM} {tempo}')
        # # Envia a mensagem ao servidor que retorna a resposta
        # resposta = client.get_fit_response(MENSAGEM)
        # print(f'Servidor -> {resposta.mensagem} {resposta.tempo}')
        # print(f'Duração Ping -> Pong: {time.time() - tempo}')
        # print('--------------------------------')
        # time.sleep(1)
    client.get_fit_response()
    