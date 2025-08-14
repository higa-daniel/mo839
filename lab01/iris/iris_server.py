import grpc
from concurrent import futures
import time
from sklearn.model_selection import train_test_split

import iris_ping_pong_pb2_grpc as pb2_grpc
import iris_ping_pong_pb2 as pb2

# x_treino, x_teste, y_treino, y_test = train_test_split(atributos, rotulos, test_size=0.2)

class IrisServer(pb2_grpc.IrisPingPongService):

    def GetServerResponseFit(self, request, context):
        mensagem = request.mensagem
        resposta = f"Pong!"

        mensagem_resposta = {
            'mensagem' : resposta,
            'tempo'    : time.time()
        }

        return pb2.Pong(**mensagem_resposta)
    
    def GetServerResponsePredict(self, request, context):
        mensagem = request.mensagem
        resposta = f"Pong!"

        mensagem_resposta = {
            'mensagem' : resposta,
            'tempo'    : time.time()
        }

        return pb2.Pong(**mensagem_resposta)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_IrisPingPongService_to_server(IrisServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()