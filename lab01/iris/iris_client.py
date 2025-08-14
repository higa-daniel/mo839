import grpc
import iris_ping_pong_pb2 as iris_pb2
import iris_ping_pong_pb2_grpc as iris_pb2_grpc
import time
from sklearn.datasets import load_iris

class IrisClient(object):
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub        = iris_pb2_grpc.IrisPingPongStub(self.channel)
    def get_fit_response(self, message):
        message = iris_pb2_grpc.FitRequest
        iris = load_iris()
        atributos = iris.data
        rotulos = iris.target
        return self.stub.GetServerResponseFit(msg=message)
    
    def get_pred_response(self, message):
        message = iris_pb2_grpc.PredRequest
        return self.stub.GetServerResponsePredict(msg=message)
   

    if __name__ == '__main__':
           client   = IrisClient()
    while True:
        print(f'Cliente -> {msg}')
        response = client.get_fit_response(msg)
