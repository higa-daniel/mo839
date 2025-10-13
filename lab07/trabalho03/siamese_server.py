"""
Siamese Server
"""
from concurrent import futures
import time
import grpc
import numpy as np
import siamese_pb2
import siamese_pb2_grpc


class SiameseServicer(siamese_pb2_grpc.SiameseServiceServicer):
    """Implementação do servidor gRPC para comparação de similaridade"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.comparison_count = 0

    def euclidean_distance(self, vec1, vec2):
        """Calcula a distância euclidiana entre dois vetores"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def CompareSimilarity(self, request, context):
        """Compara dois embeddings e retorna a similaridade"""
        try:
            # Extrair embeddings
            embedding1 = list(request.embedding1.vector)
            embedding2 = list(request.embedding2.vector)
            client1_id = request.embedding1.client_id
            client2_id = request.embedding2.client_id

            # Calcular distância
            distance = self.euclidean_distance(embedding1, embedding2)

            # Determinar se são similares
            is_similar = distance < self.threshold

            self.comparison_count += 1

            print(f"\n{'='*60}")
            print(f"Comparação #{self.comparison_count}")
            print(f"Cliente 1: {client1_id}")
            print(f"Cliente 2: {client2_id}")
            print(f"Distância Euclidiana: {distance:.4f}")
            print(f"Similar: {'SIM' if is_similar else 'NÃO'} (threshold={self.threshold})")
            print(f"{'='*60}\n")

            # Criar resposta
            response = siamese_pb2.SimilarityResponse(
                distance=float(distance),
                is_similar=is_similar,
                client1_id=client1_id,
                client2_id=client2_id
            )

            return response

        except Exception as e:
            print(f"Erro ao processar comparação: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Erro interno: {str(e)}")
            return siamese_pb2.SimilarityResponse()

def serve(port=50051, threshold=0.5):
    """Inicia o servidor gRPC"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    siamese_pb2_grpc.add_SiameseServiceServicer_to_server(
        SiameseServicer(threshold=threshold),
        server
    )

    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"\n{'='*60}")
    print(f"Servidor gRPC iniciado na porta {port}")
    print(f"Threshold de similaridade: {threshold}")
    print(f"Aguardando conexões de clientes...")
    print(f"{'='*60}\n")

    try:
        while True:
            time.sleep(86400)  # Manter servidor ativo
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        server.stop(0)

if __name__ == '__main__':
    serve(port=50051, threshold=0.5)
    