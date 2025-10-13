import grpc
import numpy as np
import time
import argparse
import sys
import os

# Suprimir avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

# Importar os arquivos gerados pelo protoc
try:
    import siamese_pb2
    import siamese_pb2_grpc
except ImportError:
    print("ERRO: Arquivos gRPC não encontrados!")
    print("Execute: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. siamese.proto")
    sys.exit(1)


class SiameseClient:
    """Cliente para processamento local de imagens e envio de embeddings"""
    
    def __init__(self, client_id, model_path='base_network_model.h5', server_address='localhost:50051'):
        self.client_id = client_id
        self.server_address = server_address
        
        print(f"\n{'='*60}")
        print(f"Inicializando Cliente {client_id}")
        print(f"{'='*60}")
        
        # Carregar modelo
        print(f"Carregando modelo: {model_path}...")
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Modelo carregado com sucesso!")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            sys.exit(1)
        
        print(f"Servidor: {server_address}")
        
        # Carregar dados de teste
        print("Carregando dataset Fashion MNIST...")
        try:
            (_, _), (self.x_test, self.y_test) = fashion_mnist.load_data()
            self.x_test = self.x_test.astype('float32') / 255.0
            self.x_test = np.expand_dims(self.x_test, -1)
            print(f"✓ Dataset carregado: {len(self.x_test)} imagens")
        except Exception as e:
            print(f"✗ Erro ao carregar dataset: {e}")
            sys.exit(1)
        
        # Nomes das classes do Fashion MNIST
        self.class_names = [
            'Camiseta/top', 'Calça', 'Suéter', 'Vestido', 'Casaco',
            'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota'
        ]
        
        print(f"{'='*60}\n")
    
    def get_random_image(self):
        """Seleciona uma imagem aleatória do dataset"""
        idx = np.random.randint(0, len(self.x_test))
        image = self.x_test[idx]
        label = self.y_test[idx]
        return image, label, idx
    
    def get_embedding(self, image):
        """Processa a imagem localmente e retorna o embedding"""
        image_batch = np.expand_dims(image, 0)
        embedding = self.model.predict(image_batch, verbose=0)[0]
        return embedding
    
    def create_embedding_message(self, embedding):
        """Cria mensagem de embedding"""
        embedding_msg = siamese_pb2.Embedding(
            vector=embedding.tolist(),
            client_id=self.client_id
        )
        return embedding_msg
    
    def compare_with_server(self, embedding1_msg, embedding2_msg):
        """Envia dois embeddings para o servidor comparar"""
        try:
            with grpc.insecure_channel(self.server_address) as channel:
                stub = siamese_pb2_grpc.SiameseServiceStub(channel)
                
                pair = siamese_pb2.EmbeddingPair(
                    embedding1=embedding1_msg,
                    embedding2=embedding2_msg
                )
                
                response = stub.CompareSimilarity(pair)
                return response
        except grpc.RpcError as e:
            print(f"\n✗ Erro de comunicação com servidor: {e.code()}")
            print(f"   Detalhes: {e.details()}")
            print(f"\n⚠ Certifique-se de que o servidor está rodando:")
            print(f"   python siamese_server.py")
            return None
    
    def run_single_comparison(self):
        """Executa uma única comparação"""
        # Selecionar imagem aleatória
        image, label, idx = self.get_random_image()
        
        print(f"\nCliente {self.client_id}:")
        print(f"  - Imagem selecionada: índice {idx}")
        print(f"  - Classe: {self.class_names[label]}")
        
        # Gerar embedding
        print(f"  - Gerando embedding...")
        embedding = self.get_embedding(image)
        print(f"  - Embedding gerado: vetor de dimensão {len(embedding)}")
        
        # Criar mensagem
        embedding_msg = self.create_embedding_message(embedding)
        
        return embedding_msg, image, label


def run_demo(server_address='localhost:50051', num_comparisons=5):
    """Demonstração com múltiplas comparações"""
    import matplotlib
    matplotlib.use('TkAgg')  # Backend para visualização
    import matplotlib.pyplot as plt
    
    print(f"\n{'#'*60}")
    print(f"DEMONSTRAÇÃO DE REDE SIAMESA DISTRIBUÍDA")
    print(f"{'#'*60}\n")
    
    # Verificar se servidor está acessível
    print("Verificando conexão com servidor...")
    try:
        channel = grpc.insecure_channel(server_address)
        grpc.channel_ready_future(channel).result(timeout=5)
        channel.close()
        print("✓ Servidor acessível!\n")
    except grpc.FutureTimeoutError:
        print(f"\n✗ ERRO: Não foi possível conectar ao servidor em {server_address}")
        print("\n⚠ Certifique-se de que o servidor está rodando:")
        print("   python siamese_server.py")
        sys.exit(1)
    
    # Criar dois clientes
    print("Criando clientes...\n")
    client1 = SiameseClient('Cliente-A', server_address=server_address)
    client2 = SiameseClient('Cliente-B', server_address=server_address)
    
    time.sleep(1)
    
    for i in range(num_comparisons):
        print(f"\n{'='*60}")
        print(f"COMPARAÇÃO {i+1}/{num_comparisons}")
        print(f"{'='*60}\n")
        
        # Cliente 1 processa imagem
        emb1_msg, img1, label1 = client1.run_single_comparison()
        
        # Cliente 2 processa imagem
        emb2_msg, img2, label2 = client2.run_single_comparison()
        
        # Enviar para servidor comparar
        print("\nEnviando embeddings para o servidor...")
        response = client1.compare_with_server(emb1_msg, emb2_msg)
        
        if response is None:
            print("✗ Falha na comunicação com servidor")
            continue
        
        # Mostrar resultado
        print(f"\n{'='*60}")
        print(f"RESULTADO DA COMPARAÇÃO:")
        print(f"{'='*60}")
        print(f"  - Distância: {response.distance:.4f}")
        print(f"  - Similar: {'SIM' if response.is_similar else 'NÃO'}")
        print(f"  - Classes: {client1.class_names[label1]} vs {client2.class_names[label2]}")
        print(f"  - Classes iguais: {'SIM' if label1 == label2 else 'NÃO'}")
        print(f"{'='*60}\n")
        
        # Visualizar
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img1.squeeze(), cmap='gray')
        axes[0].set_title(f'Cliente A\n{client1.class_names[label1]}', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(img2.squeeze(), cmap='gray')
        axes[1].set_title(f'Cliente B\n{client2.class_names[label2]}', fontsize=12)
        axes[1].axis('off')
        
        color = 'green' if response.is_similar else 'red'
        result_text = f"Distância: {response.distance:.4f}\n"
        result_text += f"Similar: {'SIM' if response.is_similar else 'NÃO'}\n"
        result_text += f"Classes iguais: {'SIM' if label1 == label2 else 'NÃO'}"
        
        axes[2].text(0.5, 0.5, result_text,
                    ha='center', va='center',
                    fontsize=14, color=color, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'comparison_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        
        time.sleep(1)
    
    print(f"\n{'#'*60}")
    print(f"DEMONSTRAÇÃO CONCLUÍDA")
    print(f"{'#'*60}\n")


def run_interactive_mode(server_address='localhost:50051'):
    """Modo interativo para enviar embeddings"""
    client_id = input("Digite o ID do cliente: ").strip()
    
    if not client_id:
        client_id = f"Cliente-{np.random.randint(1000, 9999)}"
        print(f"Usando ID gerado: {client_id}")
    
    # Verificar se servidor está acessível
    print("\nVerificando conexão com servidor...")
    try:
        channel = grpc.insecure_channel(server_address)
        grpc.channel_ready_future(channel).result(timeout=5)
        channel.close()
        print("✓ Servidor acessível!\n")
    except grpc.FutureTimeoutError:
        print(f"\n✗ ERRO: Não foi possível conectar ao servidor em {server_address}")
        print("\n⚠ Certifique-se de que o servidor está rodando:")
        print("   python siamese_server.py")
        sys.exit(1)
    
    client = SiameseClient(client_id, server_address=server_address)
    
    print("\n" + "="*60)
    print("MODO INTERATIVO")
    print("="*60)
    print("\nEste cliente irá processar imagens e gerar embeddings.")
    print("Para comparar similaridade, você precisa de outro cliente rodando.")
    print("\nPressione Ctrl+C para sair.\n")
    
    try:
        while True:
            input("Pressione Enter para processar uma imagem...")
            
            emb_msg, img, label = client.run_single_comparison()
            
            print(f"\n✓ Embedding gerado e pronto para envio!")
            print(f"  Cliente: {client_id}")
            print(f"  Classe da imagem: {client.class_names[label]}")
            print(f"  Tamanho do vetor: {len(emb_msg.vector)} dimensões")
            
            # Visualizar imagem
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(4, 4))
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'{client.class_names[label]}', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            
            print("\n" + "-"*60 + "\n")
            
    except KeyboardInterrupt:
        print(f"\n\n✓ Cliente {client_id} encerrado.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cliente da Rede Siamesa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Modo demonstração (recomendado)
  python siamese_client.py --demo --comparisons 5

  # Modo interativo
  python siamese_client.py

  # Especificar servidor
  python siamese_client.py --server localhost:50051 --demo
        """
    )
    
    parser.add_argument('--server', default='localhost:50051', 
                       help='Endereço do servidor (padrão: localhost:50051)')
    parser.add_argument('--demo', action='store_true', 
                       help='Executar demonstração automática')
    parser.add_argument('--comparisons', type=int, default=5, 
                       help='Número de comparações no modo demo (padrão: 5)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CLIENTE DE REDE SIAMESA DISTRIBUÍDA")
    print("="*60)
    
    if args.demo:
        print("\nModo: DEMONSTRAÇÃO AUTOMÁTICA")
        print(f"Servidor: {args.server}")
        print(f"Comparações: {args.comparisons}\n")
        run_demo(server_address=args.server, num_comparisons=args.comparisons)
    else:
        print("\nModo: INTERATIVO")
        print(f"Servidor: {args.server}\n")
        run_interactive_mode(server_address=args.server)