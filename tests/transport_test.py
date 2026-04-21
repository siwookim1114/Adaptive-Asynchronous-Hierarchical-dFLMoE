"""
Real TCP Transport Test
Script to demonstrate how the TCP transport is used for P2P communication.
Communication over TCP sockets
"""

from infra.transport import TCPTransport, Message
import time
import sys

def example_two_clients_same_machine():
    """Example: Two clients on same machine (different ports)"""
    print("="*70)
    print("EXAMPLE: Two Clients on Same Machine")
    print("="*70 + "\n")
    
    # Create Client A
    print("Creating Client A...")
    transport_a = TCPTransport(
        client_id="client_a",
        host="127.0.0.1",
        port=5000
    )
    
    # Create Client B
    print("Creating Client B...")
    transport_b = TCPTransport(
        client_id="client_b",
        host="127.0.0.1",
        port=5001
    )
    
    # Define message handlers
    received_a = []
    received_b = []
    
    def handle_message_a(message: Message):
        print(f"\n[Client A] Received message!")
        print(f"  From: {message.sender_id}")
        print(f"  Type: {message.message_type}")
        print(f"  Payload: {message.payload}")
        received_a.append(message)
    
    def handle_message_b(message: Message):
        print(f"\n[Client B] Received message!")
        print(f"  From: {message.sender_id}")
        print(f"  Type: {message.message_type}")
        print(f"  Payload: {message.payload}")
        received_b.append(message)
    
    # Register handlers
    transport_a.register_handler('test_message', handle_message_a)
    transport_b.register_handler('test_message', handle_message_b)
    
    # Register peers (tell each other's addresses)
    print("\nRegistering peers...")
    transport_a.register_peer("client_b", "127.0.0.1", 5001)
    transport_b.register_peer("client_a", "127.0.0.1", 5000)
    
    # Test 1: A sends to B
    print("\n" + "-"*70)
    print("Test 1: Client A → Client B")
    print("-"*70)
    payload = {'message': 'Hello from A!', 'data': [1, 2, 3, 4, 5]}
    success = transport_a.send("client_b", "test_message", payload)
    print(f"Send result: {success}")
    time.sleep(0.5)  # Wait for network
    
    # Test 2: B sends to A
    print("\n" + "-"*70)
    print("Test 2: Client B → Client A")
    print("-"*70)
    payload = {'message': 'Hello from B!', 'value': 42}
    success = transport_b.send("client_a", "test_message", payload)
    print(f"Send result: {success}")
    time.sleep(0.5)
    
    # Test 3: Multiple messages
    print("\n" + "-"*70)
    print("Test 3: Multiple Messages")
    print("-"*70)
    for i in range(3):
        payload = {'sequence': i, 'message': f'Message {i} from A'}
        transport_a.send("client_b", "test_message", payload)
    time.sleep(0.5)
    
    # Print statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    transport_a.print_statistics()
    transport_b.print_statistics()
    
    # Verify
    print("="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"Client A received {len(received_a)} messages")
    print(f"Client B received {len(received_b)} messages")
    
    if len(received_a) > 0 and len(received_b) > 0:
        print("\nSUCCESS: TCP communication working!")
    else:
        print("\nFAILURE: No messages received")
    
    # Cleanup
    print("\nShutting down...")
    transport_a.shutdown()
    transport_b.shutdown()
    
    print("\nExample complete!\n")


def example_broadcast():
    """Example: Broadcasting to multiple clients"""
    print("="*70)
    print("EXAMPLE: Broadcasting to Multiple Clients")
    print("="*70 + "\n")
    
    # Create 3 clients
    transports = []
    received_messages = {f"client_{i}": [] for i in range(3)}
    
    for i in range(3):
        client_id = f"client_{i}"
        port = 5000 + i
        
        transport = TCPTransport(
            client_id=client_id,
            host="127.0.0.1",
            port=port
        )
        
        # Define handler
        def make_handler(cid):
            def handler(message):
                print(f"[{cid}] Received '{message.message_type}' from {message.sender_id}")
                received_messages[cid].append(message)
            return handler
        
        transport.register_handler('broadcast', make_handler(client_id))
        transports.append(transport)
    
    # Register all peers with each other
    print("Registering peers...")
    for i, transport in enumerate(transports):
        for j in range(3):
            if i != j:
                transport.register_peer(f"client_{j}", "127.0.0.1", 5000 + j)
    
    # Client 0 broadcasts to everyone
    print("\n" + "-"*70)
    print("Client 0 broadcasting to all peers...")
    print("-"*70)
    peer_ids = ["client_1", "client_2"]
    payload = {'announcement': 'Hello everyone!', 'from': 'client_0'}
    num_sent = transports[0].broadcast(peer_ids, 'broadcast', payload)
    print(f"Broadcast sent to {num_sent}/{len(peer_ids)} peers")
    time.sleep(0.5)
    
    # Verify
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    for i, messages in enumerate(received_messages.values()):
        print(f"Client {i} received {len(messages)} messages")
    
    # Cleanup
    print("\nShutting down...")
    for transport in transports:
        transport.shutdown()
    
    print("\nExample complete!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REAL TCP TRANSPORT EXAMPLES")
    print("="*70 + "\n")
    
    try:
        # Example 1
        example_two_clients_same_machine()
        
        time.sleep(1)
        
        # Example 2
        example_broadcast()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*70)
    print("All examples complete!")
    print("="*70)