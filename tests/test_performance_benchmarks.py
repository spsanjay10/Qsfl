"""
Performance Benchmarking Tests for QSFL-CAAD System

Comprehensive performance tests that validate system scalability, throughput,
latency, and resource utilization under various load conditions.
"""

import pytest
import numpy as np
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import system components
from pq_security.manager import PQCryptoManager
from pq_security.kyber import KyberKeyExchange
from pq_security.dilithium import DilithiumSigner
from auth.authentication_service import AuthenticationService
from auth.credential_manager import ClientCredentialManager
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.feature_extractor import FeatureExtractor
from anomaly_detection.reputation_manager import ClientReputationManager
from federated_learning.server import SecureFederatedServer
from federated_learning.model_aggregator import FederatedAveragingAggregator

# Import test utilities
from tests.test_utils import (
    TestEnvironmentManager, ModelUpdateGenerator, MockServiceFactory,
    ExperimentRunner, create_test_model_shape
)
from anomaly_detection.interfaces import ModelUpdate


class PerformanceMonitor:
    """Monitors system performance metrics during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        
        def monitor():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_info.percent)
                    
                    time.sleep(0.5)
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring and return results."""
        self.end_time = time.time()
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return {
            'duration': self.end_time - self.start_time,
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'max_cpu_percent': np.max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_memory_percent': np.mean(self.memory_samples) if self.memory_samples else 0,
            'max_memory_percent': np.max(self.memory_samples) if self.memory_samples else 0,
            'cpu_samples': len(self.cpu_samples),
            'memory_samples': len(self.memory_samples)
        }


class TestCryptographicPerformance:
    """Test performance of cryptographic operations."""
    
    @pytest.fixture
    def pq_crypto_manager(self):
        """Create PQ crypto manager for testing."""
        kyber = KyberKeyExchange()
        dilithium = DilithiumSigner()
        return PQCryptoManager(kyber, dilithium)
    
    def test_key_generation_performance(self, pq_crypto_manager):
        """Test key generation performance."""
        num_keypairs = 100
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        # Generate multiple keypairs
        keypairs = []
        for i in range(num_keypairs):
            public_key, private_key = pq_crypto_manager.generate_keypair()
            keypairs.append((public_key, private_key))
        
        end_time = time.time()
        perf_stats = monitor.stop_monitoring()
        
        # Performance assertions
        total_time = end_time - start_time
        avg_time_per_keypair = total_time / num_keypairs
        
        assert avg_time_per_keypair < 0.1, f"Key generation too slow: {avg_time_per_keypair:.4f}s per keypair"
        assert total_time < 10.0, f"Total key generation time too slow: {total_time:.2f}s"
        
        # Verify all keypairs were generated
        assert len(keypairs) == num_keypairs
        
        return {
            'num_keypairs': num_keypairs,
            'total_time': total_time,
            'avg_time_per_keypair': avg_time_per_keypair,
            'keypairs_per_second': num_keypairs / total_time,
            'performance_stats': perf_stats
        }
    
    def test_signature_performance(self, pq_crypto_manager):
        """Test signature generation and verification performance."""
        # Generate test keypair
        public_key, private_key = pq_crypto_manager.generate_keypair()
        
        # Test data
        test_messages = [
            b"small_message",
            b"medium_message_" * 10,
            b"large_message_" * 100
        ]
        
        results = {}
        
        for msg_type, message in zip(["small", "medium", "large"], test_messages):
            num_operations = 50
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Test signing
            sign_start = time.time()
            signatures = []
            for i in range(num_operations):
                signature = pq_crypto_manager.sign(message, private_key)
                signatures.append(signature)
            sign_end = time.time()
            
            # Test verification
            verify_start = time.time()
            verifications = []
            for signature in signatures:
                is_valid = pq_crypto_manager.verify(message, signature, public_key)
                verifications.append(is_valid)
            verify_end = time.time()
            
            perf_stats = monitor.stop_monitoring()
            
            # Calculate metrics
            sign_time = sign_end - sign_start
            verify_time = verify_end - verify_start
            
            results[msg_type] = {
                'message_size': len(message),
                'num_operations': num_operations,
                'sign_time': sign_time,
                'verify_time': verify_time,
                'avg_sign_time': sign_time / num_operations,
                'avg_verify_time': verify_time / num_operations,
                'signs_per_second': num_operations / sign_time,
                'verifications_per_second': num_operations / verify_time,
                'all_verified': all(verifications),
                'performance_stats': perf_stats
            }
            
            # Performance assertions
            assert results[msg_type]['avg_sign_time'] < 0.05, \
                f"Signing too slow for {msg_type}: {results[msg_type]['avg_sign_time']:.4f}s"
            assert results[msg_type]['avg_verify_time'] < 0.05, \
                f"Verification too slow for {msg_type}: {results[msg_type]['avg_verify_time']:.4f}s"
            assert results[msg_type]['all_verified'], f"Not all signatures verified for {msg_type}"
        
        return results
    
    def test_concurrent_crypto_operations(self, pq_crypto_manager):
        """Test cryptographic operations under concurrent load."""
        num_threads = 4
        operations_per_thread = 25
        
        # Generate test keypairs
        keypairs = []
        for i in range(num_threads):
            public_key, private_key = pq_crypto_manager.generate_keypair()
            keypairs.append((public_key, private_key))
        
        def crypto_worker(thread_id: int, keypair: Tuple[bytes, bytes]) -> Dict[str, Any]:
            """Worker function for concurrent crypto operations."""
            public_key, private_key = keypair
            message = f"thread_{thread_id}_message".encode()
            
            start_time = time.time()
            
            # Perform signing and verification operations
            for i in range(operations_per_thread):
                signature = pq_crypto_manager.sign(message, private_key)
                is_valid = pq_crypto_manager.verify(message, signature, public_key)
                
                if not is_valid:
                    return {'success': False, 'error': f'Verification failed at operation {i}'}
            
            end_time = time.time()
            
            return {
                'success': True,
                'thread_id': thread_id,
                'operations': operations_per_thread,
                'duration': end_time - start_time,
                'ops_per_second': operations_per_thread / (end_time - start_time)
            }
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Run concurrent operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(crypto_worker, i, keypairs[i])
                for i in range(num_threads)
            ]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        perf_stats = monitor.stop_monitoring()
        
        # Verify all operations succeeded
        successful_threads = sum(1 for result in results if result['success'])
        assert successful_threads == num_threads, f"Only {successful_threads}/{num_threads} threads succeeded"
        
        # Calculate aggregate metrics
        total_operations = num_threads * operations_per_thread
        total_time = end_time - start_time
        aggregate_ops_per_second = total_operations / total_time
        
        return {
            'num_threads': num_threads,
            'operations_per_thread': operations_per_thread,
            'total_operations': total_operations,
            'total_time': total_time,
            'aggregate_ops_per_second': aggregate_ops_per_second,
            'thread_results': results,
            'performance_stats': perf_stats
        }


class TestAnomalyDetectionPerformance:
    """Test performance of anomaly detection components."""
    
    @pytest.fixture
    def detection_components(self):
        """Set up anomaly detection components."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        return detector, feature_extractor
    
    def test_training_performance(self, detection_components):
        """Test anomaly detector training performance."""
        detector, feature_extractor = detection_components
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("medium")
        
        # Test different training set sizes
        training_sizes = [50, 100, 200, 500]
        results = {}
        
        for size in training_sizes:
            # Generate training data
            training_updates = []
            for i in range(size):
                update = generator.generate_honest_update(
                    f"client_{i}", "training", model_shape
                )
                training_updates.append(update)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Train detector
            start_time = time.time()
            detector.fit(training_updates)
            end_time = time.time()
            
            perf_stats = monitor.stop_monitoring()
            
            training_time = end_time - start_time
            
            results[size] = {
                'training_size': size,
                'training_time': training_time,
                'samples_per_second': size / training_time,
                'performance_stats': perf_stats
            }
            
            # Performance assertions
            assert training_time < 30.0, f"Training too slow for size {size}: {training_time:.2f}s"
            assert results[size]['samples_per_second'] > 1.0, \
                f"Training throughput too low: {results[size]['samples_per_second']:.2f} samples/s"
        
        return results
    
    def test_scoring_performance(self, detection_components):
        """Test anomaly scoring performance."""
        detector, feature_extractor = detection_components
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Train detector
        training_updates = []
        for i in range(100):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            training_updates.append(update)
        detector.fit(training_updates)
        
        # Test scoring performance
        num_test_updates = 1000
        test_updates = []
        
        for i in range(num_test_updates):
            if i % 10 == 0:  # 10% malicious
                update = generator.generate_malicious_update(
                    f"test_{i}", "test", model_shape, "gradient_poisoning", 2.0
                )
            else:  # 90% honest
                update = generator.generate_honest_update(f"test_{i}", "test", model_shape)
            test_updates.append(update)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Score all updates
        start_time = time.time()
        scores = []
        for update in test_updates:
            score = detector.predict_anomaly_score(update)
            scores.append(score)
        end_time = time.time()
        
        perf_stats = monitor.stop_monitoring()
        
        # Calculate metrics
        scoring_time = end_time - start_time
        scores_per_second = num_test_updates / scoring_time
        avg_score_time = scoring_time / num_test_updates
        
        # Performance assertions
        assert avg_score_time < 0.01, f"Scoring too slow: {avg_score_time:.4f}s per update"
        assert scores_per_second > 100, f"Scoring throughput too low: {scores_per_second:.2f} scores/s"
        
        # Verify scores are reasonable
        assert all(0.0 <= score <= 1.0 for score in scores), "Scores out of valid range"
        
        return {
            'num_updates': num_test_updates,
            'scoring_time': scoring_time,
            'scores_per_second': scores_per_second,
            'avg_score_time': avg_score_time,
            'score_distribution': {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            },
            'performance_stats': perf_stats
        }
    
    def test_concurrent_scoring(self, detection_components):
        """Test concurrent anomaly scoring performance."""
        detector, feature_extractor = detection_components
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Train detector
        training_updates = []
        for i in range(50):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            training_updates.append(update)
        detector.fit(training_updates)
        
        # Prepare test updates
        num_threads = 4
        updates_per_thread = 100
        
        def scoring_worker(thread_id: int) -> Dict[str, Any]:
            """Worker function for concurrent scoring."""
            thread_updates = []
            for i in range(updates_per_thread):
                update = generator.generate_honest_update(
                    f"thread_{thread_id}_client_{i}", "test", model_shape
                )
                thread_updates.append(update)
            
            start_time = time.time()
            scores = []
            for update in thread_updates:
                score = detector.predict_anomaly_score(update)
                scores.append(score)
            end_time = time.time()
            
            return {
                'thread_id': thread_id,
                'num_updates': len(thread_updates),
                'duration': end_time - start_time,
                'scores_per_second': len(thread_updates) / (end_time - start_time),
                'avg_score': np.mean(scores)
            }
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Run concurrent scoring
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(scoring_worker, i) for i in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        perf_stats = monitor.stop_monitoring()
        
        # Calculate aggregate metrics
        total_updates = num_threads * updates_per_thread
        total_time = end_time - start_time
        aggregate_scores_per_second = total_updates / total_time
        
        # Performance assertions
        assert aggregate_scores_per_second > 50, \
            f"Concurrent scoring too slow: {aggregate_scores_per_second:.2f} scores/s"
        
        return {
            'num_threads': num_threads,
            'updates_per_thread': updates_per_thread,
            'total_updates': total_updates,
            'total_time': total_time,
            'aggregate_scores_per_second': aggregate_scores_per_second,
            'thread_results': results,
            'performance_stats': perf_stats
        }


class TestFederatedLearningPerformance:
    """Test performance of federated learning operations."""
    
    @pytest.fixture
    def federated_server(self):
        """Create federated learning server for testing."""
        # Use mock services for performance testing
        mock_auth = MockServiceFactory.create_mock_auth_service()
        mock_pq = MockServiceFactory.create_mock_pq_crypto()
        mock_detector = MockServiceFactory.create_mock_anomaly_detector()
        mock_reputation = MockServiceFactory.create_mock_reputation_manager()
        
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=mock_auth,
            pq_crypto=mock_pq,
            aggregator=aggregator,
            anomaly_detector=mock_detector,
            reputation_manager=mock_reputation
        )
        
        return server, mock_auth
    
    def test_single_round_performance(self, federated_server):
        """Test performance of single federated learning round."""
        server, mock_auth = federated_server
        generator = ModelUpdateGenerator(seed=42)
        
        # Test different client counts
        client_counts = [5, 10, 25, 50, 100]
        results = {}
        
        for num_clients in client_counts:
            model_shape = create_test_model_shape("small")
            
            # Initialize server
            initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                             for name, shape in model_shape.items()}
            server.initialize_global_model(initial_weights)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Start round
            round_start = time.time()
            round_id = server.start_training_round()
            
            # Generate and submit updates
            updates_start = time.time()
            for i in range(num_clients):
                client_id = f"client_{i:03d}"
                mock_auth.register_client(client_id)
                
                update = generator.generate_honest_update(client_id, round_id, model_shape)
                server.receive_client_update(client_id, update)
            updates_end = time.time()
            
            # Aggregate
            aggregation_start = time.time()
            global_model = server.aggregate_updates(round_id)
            aggregation_end = time.time()
            
            round_end = time.time()
            perf_stats = monitor.stop_monitoring()
            
            # Calculate metrics
            total_round_time = round_end - round_start
            updates_time = updates_end - updates_start
            aggregation_time = aggregation_end - aggregation_start
            
            results[num_clients] = {
                'num_clients': num_clients,
                'total_round_time': total_round_time,
                'updates_time': updates_time,
                'aggregation_time': aggregation_time,
                'updates_per_second': num_clients / updates_time,
                'clients_per_second': num_clients / total_round_time,
                'aggregation_success': global_model is not None,
                'performance_stats': perf_stats
            }
            
            # Performance assertions
            assert total_round_time < 10.0, \
                f"Round too slow for {num_clients} clients: {total_round_time:.2f}s"
            assert results[num_clients]['updates_per_second'] > 10, \
                f"Update processing too slow: {results[num_clients]['updates_per_second']:.2f} updates/s"
        
        return results
    
    def test_multiple_rounds_performance(self, federated_server):
        """Test performance across multiple federated learning rounds."""
        server, mock_auth = federated_server
        generator = ModelUpdateGenerator(seed=42)
        
        num_clients = 20
        num_rounds = 10
        model_shape = create_test_model_shape("medium")
        
        # Initialize server
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Register clients once
        for i in range(num_clients):
            client_id = f"client_{i:03d}"
            mock_auth.register_client(client_id)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        round_times = []
        aggregation_times = []
        
        total_start = time.time()
        
        for round_num in range(num_rounds):
            round_start = time.time()
            
            # Start round
            round_id = server.start_training_round()
            
            # Submit updates
            for i in range(num_clients):
                client_id = f"client_{i:03d}"
                update = generator.generate_honest_update(client_id, round_id, model_shape)
                server.receive_client_update(client_id, update)
            
            # Aggregate
            agg_start = time.time()
            global_model = server.aggregate_updates(round_id)
            agg_end = time.time()
            
            round_end = time.time()
            
            round_times.append(round_end - round_start)
            aggregation_times.append(agg_end - agg_start)
            
            assert global_model is not None, f"Aggregation failed in round {round_num}"
        
        total_end = time.time()
        perf_stats = monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = total_end - total_start
        avg_round_time = np.mean(round_times)
        avg_aggregation_time = np.mean(aggregation_times)
        
        # Performance assertions
        assert avg_round_time < 2.0, f"Average round time too slow: {avg_round_time:.2f}s"
        assert total_time < 30.0, f"Total training time too slow: {total_time:.2f}s"
        
        return {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'total_time': total_time,
            'avg_round_time': avg_round_time,
            'avg_aggregation_time': avg_aggregation_time,
            'rounds_per_minute': (num_rounds * 60) / total_time,
            'round_times': round_times,
            'aggregation_times': aggregation_times,
            'performance_stats': perf_stats
        }
    
    def test_large_model_performance(self, federated_server):
        """Test performance with large model sizes."""
        server, mock_auth = federated_server
        generator = ModelUpdateGenerator(seed=42)
        
        # Test different model sizes
        model_sizes = {
            'small': create_test_model_shape("small"),
            'medium': create_test_model_shape("medium"),
            'large': {
                'layer1': (500, 250),
                'layer2': (250, 100),
                'layer3': (100, 50),
                'layer4': (50, 10),
                'layer5': (10, 1)
            }
        }
        
        num_clients = 10
        results = {}
        
        for size_name, model_shape in model_sizes.items():
            # Initialize server
            initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                             for name, shape in model_shape.items()}
            server.initialize_global_model(initial_weights)
            
            # Register clients
            for i in range(num_clients):
                client_id = f"client_{i:03d}"
                mock_auth.register_client(client_id)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Run single round
            start_time = time.time()
            round_id = server.start_training_round()
            
            # Submit updates
            for i in range(num_clients):
                client_id = f"client_{i:03d}"
                update = generator.generate_honest_update(client_id, round_id, model_shape)
                server.receive_client_update(client_id, update)
            
            # Aggregate
            global_model = server.aggregate_updates(round_id)
            end_time = time.time()
            
            perf_stats = monitor.stop_monitoring()
            
            # Calculate model size metrics
            total_params = sum(np.prod(shape) for shape in model_shape.values())
            model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            round_time = end_time - start_time
            
            results[size_name] = {
                'model_shape': model_shape,
                'total_parameters': total_params,
                'model_size_mb': model_size_mb,
                'round_time': round_time,
                'params_per_second': (total_params * num_clients) / round_time,
                'mb_per_second': (model_size_mb * num_clients) / round_time,
                'aggregation_success': global_model is not None,
                'performance_stats': perf_stats
            }
            
            # Performance assertions based on model size
            max_time = 5.0 if size_name == 'large' else 2.0
            assert round_time < max_time, \
                f"Round too slow for {size_name} model: {round_time:.2f}s"
        
        return results


class TestSystemScalability:
    """Test system scalability under increasing load."""
    
    def test_client_scalability(self):
        """Test system behavior with increasing number of clients."""
        # This would test the system with progressively more clients
        # to identify scalability limits and performance degradation points
        
        client_counts = [10, 25, 50, 100, 200, 500]
        results = {}
        
        for num_clients in client_counts:
            # Setup system components
            mock_auth = MockServiceFactory.create_mock_auth_service()
            mock_pq = MockServiceFactory.create_mock_pq_crypto()
            mock_detector = MockServiceFactory.create_mock_anomaly_detector()
            mock_reputation = MockServiceFactory.create_mock_reputation_manager()
            
            aggregator = FederatedAveragingAggregator()
            server = SecureFederatedServer(
                auth_service=mock_auth,
                pq_crypto=mock_pq,
                aggregator=aggregator,
                anomaly_detector=mock_detector,
                reputation_manager=mock_reputation
            )
            
            generator = ModelUpdateGenerator(seed=42)
            model_shape = create_test_model_shape("small")
            
            # Initialize server
            initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                             for name, shape in model_shape.items()}
            server.initialize_global_model(initial_weights)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            try:
                # Test single round with many clients
                start_time = time.time()
                round_id = server.start_training_round()
                
                # Submit updates from all clients
                for i in range(num_clients):
                    client_id = f"scale_client_{i:04d}"
                    mock_auth.register_client(client_id)
                    
                    update = generator.generate_honest_update(client_id, round_id, model_shape)
                    server.receive_client_update(client_id, update)
                
                # Aggregate
                global_model = server.aggregate_updates(round_id)
                end_time = time.time()
                
                perf_stats = monitor.stop_monitoring()
                
                results[num_clients] = {
                    'num_clients': num_clients,
                    'success': True,
                    'round_time': end_time - start_time,
                    'clients_per_second': num_clients / (end_time - start_time),
                    'performance_stats': perf_stats
                }
                
            except Exception as e:
                perf_stats = monitor.stop_monitoring()
                results[num_clients] = {
                    'num_clients': num_clients,
                    'success': False,
                    'error': str(e),
                    'performance_stats': perf_stats
                }
        
        # Analyze scalability results
        successful_counts = [count for count, result in results.items() if result['success']]
        max_clients = max(successful_counts) if successful_counts else 0
        
        # Should handle at least 100 clients
        assert max_clients >= 100, f"System should scale to at least 100 clients, max: {max_clients}"
        
        return results
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with system load."""
        # This would monitor memory usage as load increases
        # to identify memory leaks or excessive memory consumption
        
        # Placeholder for memory scaling tests
        assert True
    
    def test_concurrent_round_handling(self):
        """Test system behavior with concurrent training rounds."""
        # This would test if the system can handle multiple
        # overlapping training rounds or rapid round succession
        
        # Placeholder for concurrent round tests
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])