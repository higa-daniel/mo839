"""
Iris Client with Byzantine attack simulation - FIXED VERSION
"""
import grpc
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
import time
import hashlib

import iris_pb2_grpc as pb2_grpc
import iris_pb2 as pb2

class ByzantineClient:
    """Base client class with attack capabilities"""
    
    def __init__(self, client_id, attack_type=None, attack_strength=0.3):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub = pb2_grpc.IrisStub(self.channel)
        self.model_type = "knn"
        self.client_id = client_id
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        
        # Load and split data
        iris = load_iris()
        self.full_data = iris.data
        self.full_labels = iris.target
        
        # Convert client_id to integer seed for random_state
        seed = self._client_id_to_seed(client_id)
        
        # Different data split for each client
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            self.full_data, self.full_labels, test_size=0.2, random_state=seed
        )
        
        # Apply Byzantine attack if specified
        if attack_type:
            x_treino, y_treino = self.apply_attack(x_treino, y_treino)
        
        self.x_treino = x_treino
        self.x_teste = x_teste
        self.y_treino = y_treino
        self.y_teste = y_teste
    
    def _client_id_to_seed(self, client_id):
        """Convert client_id string to integer seed"""
        # Use hash to convert string to consistent integer
        hash_obj = hashlib.md5(client_id.encode())
        return int(hash_obj.hexdigest()[:8], 16) % 10000  # Limit to reasonable range
    
    def apply_attack(self, features, labels):
        """Apply different types of Byzantine attacks"""
        if self.attack_type == "feature_noise":
            # Add random noise to all features
            noise = np.random.normal(0, self.attack_strength * np.std(features), features.shape)
            return features + noise, labels.copy()
            
        elif self.attack_type == "single_feature_noise":
            # Add noise to only one feature
            feature_idx = random.randint(0, features.shape[1] - 1)
            noise = np.random.normal(0, self.attack_strength * np.std(features[:, feature_idx]), 
                                   features.shape[0])
            features_attacked = features.copy()
            features_attacked[:, feature_idx] += noise
            return features_attacked, labels.copy()
            
        elif self.attack_type == "label_flip":
            # Flip labels randomly
            labels_attacked = labels.copy()
            flip_mask = np.random.random(len(labels)) < self.attack_strength
            unique_labels = np.unique(labels)
            for i in np.where(flip_mask)[0]:
                possible_labels = [l for l in unique_labels if l != labels[i]]
                labels_attacked[i] = random.choice(possible_labels)
            return features.copy(), labels_attacked
            
        else:
            return features.copy(), labels.copy()
    
    def train_model(self):
        """Send training request to server"""
        flattened_features = self.x_treino.flatten().tolist()
        request = pb2.FitRequest(
            features=flattened_features,
            labels=self.y_treino.tolist(),
            rows=self.x_treino.shape[0],
            cols=self.x_treino.shape[1],
            model_type=self.model_type,
            client_id=self.client_id
        )
        
        print(f"Client {self.client_id} ({self.attack_type or 'normal'}) training...")
        try:
            response = self.stub.GetServerResponseFit(request)
            
            print(f"Client {self.client_id} - {response.message}")
            if response.accuracy > 0:
                print(f"  Accuracy: {response.accuracy:.4f}, Time: {response.training_time:.4f}s")
            print("-" * 50)
            
            return response
        except grpc.RpcError as e:
            print(f"Error training client {self.client_id}: {e}")
            return None
    
    def test_model(self):
        """Test the trained model"""
        flattened_features = self.x_teste.flatten().tolist()
        request = pb2.PredictRequest(
            features=flattened_features,
            labels=self.y_teste.tolist(),
            rows=self.x_teste.shape[0],
            cols=self.x_teste.shape[1],
            model_type=self.model_type
        )
        
        try:
            response = self.stub.GetServerResponsePredict(request)
            print(f"Client {self.client_id} test - Accuracy: {response.accuracy:.4f}")
            return response
        except grpc.RpcError as e:
            print(f"Error testing client {self.client_id}: {e}")
            return None

def run_scenario(scenario_name, client_configs):
    """Run a specific Byzantine scenario"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    clients = []
    for i, config in enumerate(client_configs):
        client_id = f"client_{i+1}"
        client = ByzantineClient(
            client_id=client_id,
            attack_type=config['attack'],
            attack_strength=config.get('strength', 0.3)
        )
        clients.append(client)
        time.sleep(0.5)  # Small delay between clients
    
    # Train with all clients
    training_results = []
    for client in clients:
        result = client.train_model()
        if result:
            training_results.append(result)
        time.sleep(0.5)
    
    # Test with first client
    test_accuracy = 0.0
    if clients:
        test_result = clients[0].test_model()
        if test_result:
            test_accuracy = test_result.accuracy
    
    return test_accuracy

if __name__ == '__main__':
    # Define different scenarios
    scenarios = {
        "0 Byzantine (Baseline)": [
            {'attack': None}, {'attack': None}, {'attack': None}, 
            {'attack': None}, {'attack': None}
        ],
        "1 Byzantine (Feature Noise)": [
            {'attack': None}, {'attack': None}, {'attack': None}, 
            {'attack': None}, {'attack': 'feature_noise', 'strength': 0.5}
        ],
        "2 Byzantine (Mixed Attacks)": [
            {'attack': None}, {'attack': None}, {'attack': None},
            {'attack': 'feature_noise', 'strength': 0.4},
            {'attack': 'label_flip', 'strength': 0.4}
        ],
        "3 Byzantine (All Attack Types)": [
            {'attack': None}, {'attack': None},
            {'attack': 'feature_noise', 'strength': 0.5},
            {'attack': 'single_feature_noise', 'strength': 0.6},
            {'attack': 'label_flip', 'strength': 0.5}
        ]
    }

    results = {}

    # Wait a bit for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(3)

    for scenario_name, config in scenarios.items():
        accuracy = run_scenario(scenario_name, config)
        results[scenario_name] = accuracy
        time.sleep(2)  # Wait between scenarios

    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    for scenario, acc in results.items():
        print(f"{scenario}: {acc:.4f} accuracy")
