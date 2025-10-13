# Guia
1. Instalar Dependências
```bash
pip install -r requirements.txt
```
2. Gerar Arquivos gRPC
Execute o comando abaixo para gerar os arquivos Python a partir do arquivo .proto:
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. siamese.proto
```
Isso criará os arquivos:
siamese_pb2.py
siamese_pb2_grpc.py

3. Executar treino siamese_training.ipynb
- base_network_model.h5 (modelo base para os clientes)
- siamese_model.h5 (modelo completo)
- training_history.png - Gráfico de perda e acurácia
- model_predictions.png - Exemplos de predições
- Modelo salvo em base_network_model.h5

4. Iniciar o Servidor gRPC
```bash
python siamese_server.py
```
5. Executar Clientes
Em outro terminal, execute a demonstração automática:
```bash
python siamese_client.py --demo --comparisons 5
```