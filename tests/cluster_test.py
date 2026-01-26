"""
Cluster Manager Test
"""

from infra.cluster import ClusterManager
import torch
import time

def test_basic_clustering():
    """Test basic clustering functionality"""
    print("="*70)
    print("TEST 1: Basic Clustering")
    print("="*70 + "\n")
    
    # Create cluster manager
    cluster_manager = ClusterManager(
        num_clusters=3,
        recluster_interval=100.0
    )
    
    # Register 12 clients with different features
    print("Registering 12 clients...")
    for i in range(12):
        client_id = f"client_{i}"
        
        # Create representative features (simulated)
        # In reality, these come from your model
        if i < 4:
            # Cluster 0: Similar features around [0.1, 0.1, ...]
            features = torch.randn(512) * 0.1 + 0.1
        elif i < 8:
            # Cluster 1: Similar features around [0.5, 0.5, ...]
            features = torch.randn(512) * 0.1 + 0.5
        else:
            # Cluster 2: Similar features around [0.9, 0.9, ...]
            features = torch.randn(512) * 0.1 + 0.9
        
        # Varying trust scores
        trust_score = 0.5 + 0.5 * (i / 12)
        
        cluster_manager.register_client(
            client_id=client_id,
            features=features,
            trust_score=trust_score
        )
        print(f"  Registered {client_id} (trust={trust_score:.3f})")
    
    # Perform clustering
    print("\nPerforming clustering...")
    cluster_manager.perform_clustering()
    
    # Print results
    print("\n" + "-"*70)
    print("Clustering Results:")
    print("-"*70)
    
    for i in range(12):
        client_id = f"client_{i}"
        cluster_id = cluster_manager.get_cluster_id(client_id)
        is_head = cluster_manager.is_cluster_head(client_id)
        head_str = " (HEAD)" if is_head else ""
        print(f"  {client_id}: Cluster {cluster_id}{head_str}")
    
    # Print statistics
    cluster_manager.print_statistics()
    
    # Test communication targets
    print("-"*70)
    print("Communication Targets for client_0:")
    print("-"*70)
    targets = cluster_manager.get_communication_targets("client_0")
    print(f"  Cluster peers: {targets['cluster_peers']}")
    print(f"  Cluster heads: {targets['cluster_heads']}")
    
    print("\n✅ Test 1 Complete!\n")
    return cluster_manager


def test_hierarchical_communication(cluster_manager):
    """Test hierarchical communication pattern"""
    print("="*70)
    print("TEST 2: Hierarchical Communication")
    print("="*70 + "\n")
    
    # Simulate communication for each client
    total_within = 0
    total_across = 0
    
    for i in range(12):
        client_id = f"client_{i}"
        
        # Get communication targets
        targets = cluster_manager.get_communication_targets(client_id)
        cluster_peers = targets['cluster_peers']
        cluster_heads = targets['cluster_heads']
        
        # Count connections
        within_cluster = len(cluster_peers)
        across_cluster = len(cluster_heads) if cluster_manager.is_cluster_head(client_id) else 0
        
        total_within += within_cluster
        total_across += across_cluster
        
        print(f"{client_id}:")
        print(f"  Within cluster: {within_cluster} connections")
        print(f"  Across clusters: {across_cluster} connections")
    
    print("\n" + "-"*70)
    print("Total Communication:")
    print("-"*70)
    print(f"Within clusters: {total_within} connections")
    print(f"Across clusters: {total_across} connections")
    print(f"Total: {total_within + total_across} connections")
    
    # Compare to no clustering
    no_clustering = 12 * 11
    print(f"\nWithout clustering: {no_clustering} connections")
    reduction = (1 - (total_within + total_across) / no_clustering) * 100
    print(f"Reduction: {reduction:.1f}%")
    
    print("\n✅ Test 2 Complete!\n")


def test_dynamic_updates(cluster_manager):
    """Test dynamic client updates"""
    print("="*70)
    print("TEST 3: Dynamic Updates")
    print("="*70 + "\n")
    
    # Update client_0's features significantly
    print("Updating client_0 features...")
    old_cluster = cluster_manager.get_cluster_id("client_0")
    print(f"  Old cluster: {old_cluster}")
    
    # New features very different (should move to different cluster)
    new_features = torch.randn(512) * 0.1 + 0.9  # Like cluster 2
    cluster_manager.update_client(
        client_id="client_0",
        features=new_features
    )
    
    # Re-cluster
    print("  Re-clustering...")
    cluster_manager.perform_clustering()
    
    new_cluster = cluster_manager.get_cluster_id("client_0")
    print(f"  New cluster: {new_cluster}")
    
    if new_cluster != old_cluster:
        print("  ✅ Client moved to different cluster!")
    
    # Update trust score
    print("\nUpdating client_11 trust score...")
    old_is_head = cluster_manager.is_cluster_head("client_11")
    print(f"  Was cluster head: {old_is_head}")
    
    # Decrease trust significantly
    cluster_manager.update_client(
        client_id="client_11",
        trust_score=0.1  # Very low
    )
    
    new_is_head = cluster_manager.is_cluster_head("client_11")
    print(f"  Is cluster head now: {new_is_head}")
    
    if old_is_head and not new_is_head:
        print("  ✅ Cluster head changed due to low trust!")
    
    print("\n✅ Test 3 Complete!\n")


def test_client_removal(cluster_manager):
    """Test client removal"""
    print("="*70)
    print("TEST 4: Client Removal")
    print("="*70 + "\n")
    
    print("Removing client_0...")
    cluster_id = cluster_manager.get_cluster_id("client_0")
    print(f"  Cluster before removal: {cluster_id}")
    
    cluster_manager.remove_client("client_0")
    
    # Check it's gone
    new_cluster = cluster_manager.get_cluster_id("client_0")
    print(f"  Cluster after removal: {new_cluster}")
    
    if new_cluster is None:
        print("  ✅ Client successfully removed!")
    
    # Print updated statistics
    cluster_manager.print_statistics()
    
    print("✅ Test 4 Complete!\n")


def test_integration_example():
    """Show how to integrate with full framework"""
    print("="*70)
    print("TEST 5: Integration Example")
    print("="*70 + "\n")
    
    print("Simulating federated learning training loop...")
    print()
    
    # Setup
    cluster_manager = ClusterManager(num_clusters=3)
    
    # Register 9 clients
    for i in range(9):
        client_id = f"client_{i}"
        features = torch.randn(512)
        trust = 0.5 + 0.5 * (i / 9)
        cluster_manager.register_client(client_id, features, trust)
    
    # Initial clustering
    cluster_manager.perform_clustering()
    
    # Simulate 10 training rounds
    for round in range(10):
        print(f"Round {round + 1}:")
        
        # Each client trains and updates
        for i in range(9):
            client_id = f"client_{i}"
            
            # Simulate feature updates (data distribution changes)
            new_features = torch.randn(512) + round * 0.1
            new_trust = min(1.0, 0.5 + round * 0.05)
            
            cluster_manager.update_client(
                client_id=client_id,
                features=new_features,
                trust_score=new_trust
            )
        
        # Check if re-clustering needed
        if cluster_manager.should_recluster():
            print("  → Re-clustering triggered!")
            cluster_manager.perform_clustering()
        
        # Simulate hierarchical communication
        for i in range(9):
            client_id = f"client_{i}"
            targets = cluster_manager.get_communication_targets(client_id)
            
            # Within cluster (every round)
            num_cluster_peers = len(targets['cluster_peers'])
            
            # Across clusters (every 5 rounds)
            num_heads = 0
            if round % 5 == 0 and cluster_manager.is_cluster_head(client_id):
                num_heads = len(targets['cluster_heads'])
            
            if num_cluster_peers > 0 or num_heads > 0:
                print(f"  {client_id}: Send to {num_cluster_peers} peers, "
                      f"{num_heads} heads")
    
    # Final statistics
    print("\nFinal Statistics:")
    cluster_manager.print_statistics()
    
    print("✅ Test 5 Complete!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CLUSTER MANAGER TESTS")
    print("="*70 + "\n")
    
    try:
        # Test 1: Basic clustering
        cluster_manager = test_basic_clustering()
        time.sleep(0.5)
        
        # Test 2: Hierarchical communication
        test_hierarchical_communication(cluster_manager)
        time.sleep(0.5)
        
        # Test 3: Dynamic updates
        test_dynamic_updates(cluster_manager)
        time.sleep(0.5)
        
        # Test 4: Client removal
        test_client_removal(cluster_manager)
        time.sleep(0.5)
        
        # Test 5: Integration example
        test_integration_example()
        
        print("="*70)
        print("ALL TESTS COMPLETE! ✅")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()