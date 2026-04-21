"""
Transport Layer for Real P2P Communication

For the actual implementation benchmark, real P2P communication using TCP sockets have been implemented instead of networkless simulation.
Clients can run on different machines and communicate over actual networks.

Features:
- TCP socket-based communication (reliable, ordered delivery)
- Server Component (listens for incoming connections)
- Client component (connects to peers)
- Thread-safe operations
- Message framing (handles partial sends/receives)
- Automatic reconnection
- Real network statistics
"""
import socket
import pickle
import threading
import struct
import time
import json
from typing import Dict, List, Callable, Optional, Any, Tuple

class Message:
    """
    Network message container

    Attributes:
        sender_id: ID of the client sending this message
        receiver_id: ID of the client receiving this message
        message_type: Type of message (e.g., "expert_package")
        payload: The actual data being sent
        timestamp: When the mesage was created
        message_id: Unique identifier for this message
    """
    def __init__(self, sender_id: str, receiver_id: str, message_type: str, payload: Any, timestamp: float, message_id: str):
        """Initialize message"""
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.payload = payload
        self.timestamp = timestamp
        self.message_id = message_id

    def to_bytes(self) -> bytes:
        """Serialize mesage to bytes (Object -> Bytes)"""
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes"""
        return pickle.loads(data)

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"Message(from={self.sender_id}, to={self.receiver_id}, "
                f"type={self.message_type}, id={self.message_id})")

class TCPTransport:
    """
    Real TCP-based P2P transport

    This transport uses actual TCP sockets to communicate between clients.
    Each client runs a server (listens for incoming connections) and can connect to other clients as a client (outgoing connections).

    Architecture:
    - Server Thread: Listens for incoming connections, spawns handler threads
    - Handler Threads: One per incoming connection, receives messages
    - Send: Directly sends over established connections

    Message Protocol:
    [4 bytes: message length][N bytes: pickled message]
    This framing ensures we can handle partial sends/receives correctly.
    """
    def __init__(self, client_id: str, host: str = "0.0.0.0", port: int = 0, max_connections: int = 100):  # 0 = auto-assign port
        """
        Initialize TCP transport

        Args:
            client_id: This client's unique identifier
            host: Host to bind server to ('0.0.0.0' = all interfaces)
            port: Port to bind server to (0 = auto-assign)
            max_connections: Maximum number of simultaneous connections
        """
        self.client_id = client_id
        self.host = host
        self.port = port
        self.max_connections = max_connections

        # Message handlers: {message_type: handler_function}
        self.message_handlers: Dict[str, Callable] = {}
        self.handler_lock = threading.RLock()

        # Peer connections: {peer_id: socket}
        self.peer_sockets: Dict[str, socket.socket] = {}
        self.peer_lock = threading.RLock()

        # Peer addresses: {peer_id: (host, port)}
        self.peer_addresses: Dict[str, Tuple[str, int]] = {}

        # Per-peer send locks: serializes send() calls to the same peer
        # so that length-prefixed messages don't interleave on the socket
        self.send_locks: Dict[str, threading.Lock] = {}

        # Server socket
        self.server_socket: Optional[socket.socket] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        # Statistics
        self.stats = {
            "sent": 0,
            "received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connections_accepted": 0,
            "connections_failed": 0,
            "errors": 0
        }
        self.stats_lock = threading.Lock()
        
        # Starting server
        self.start_server()
        print(f"[Transport:{client_id}] Started TCP server on {host}:{self.port}")

    def start_server(self):
        """Start TCP server to listen for incoming connections"""
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)        # IPV4, TCP
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)      # Reuse address immediately even after reconnecion

        # Bind to address
        self.server_socket.bind((self.host, self.port))

        # Get actual port if auto-assigned
        self.port = self.server_socket.getsockname()[1]      

        # Start Listening
        self.server_socket.listen(self.max_connections)

        # Start accept thread
        self.running = True
        self.server_thread = threading.Thread(
            target = self.accept_loop,               # Function to run in thread
            daemon = True,                          # Thread dies when main program exits
            name = f"Server-{self.client_id}"        # Name for debugging that will be shown in thread list
        )
        self.server_thread.start()         # Starts the thread (begins running _accept_loop in background) -> After this line: Main Thread: Returns from __init__ -> Send messages, Server Thread: Runs _accept_loop() -> Waits for connections -> Accepts connections

    def accept_loop(self):
        """Accept incoming connections (runs in background thread)"""      # Continuously accepts new connections
        print(f"[Transport:{self.client_id}] Listening for connections...")

        while self.running:      # Loop keeps running until running becomes False
            try:
                # Accept connection (blocking, but with timeout)
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, addr = self.server_socket.accept()
                except socket.timeout:
                    continue

                print(f"[Transport:{self.client_id}] Accepted connection from {addr}")

                # Update stats
                with self.stats_lock:
                    self.stats["connections_accepted"] += 1

                # Spawn handler thread for this connection -> Creates a new thread to handle each of the new accepted incoming connection. 
                """
                Threading model:

                Server Thread:
                - accept() -> connection 1      
                - Spawn handler thread 1              Handler Thread 1: Handles connection 1, Receives Messages
                - accept() -> connection 2
                - Spawn handler thread 2              Handler Thread 2: Handles connection 2, Receives Messages
                ...
                """
                handler_thread = threading.Thread(
                    target = self.handle_connection,
                    args = (client_socket, addr),
                    daemon = True,
                    name = f"Handler-{self.client_id}=={addr}"   
                )
                handler_thread.start()
            
            except Exception as e:
                if self.running:    # Only log if not shutting down
                    print(f"[Transport:{self.client_id}] Accept error: {e}")
    
    def handle_connection(self, client_socket: socket.socket, addr: Tuple):    # Receives messages from ONE peer connection -> Runs on Handler Thread (one per connection)
        """Handle incoming connection (receives messages)"""
        try:
            while self.running:          
                # Receive message length (4 bytes)
                length_data = self.recv_exact(client_socket, 4)     # Receive the first 4 bytes which contains message length -> Rest are the actual message data
                if not length_data:
                    break
                message_length = struct.unpack('!I', length_data)[0]   # Converting 4 bytes into an integer

                # Receive message data -> Receive exactly message_length bytes (the actual message)
                message_data = self.recv_exact(client_socket, message_length)   # Actual message length
                if not message_data: 
                    break

                # Deserialize message     
                message = Message.from_bytes(message_data)      # Converting bytes back into Message object

                # Update statistics
                with self.stats_lock:
                    self.stats["received"] += 1
                    self.stats["bytes_received"] += len(message_data)
                
                print(f"[Transport:{self.client_id}]Received '{message.message_type}' "
                      f"from {message.sender_id} ({len(message_data)} bytes)")
                
                # Dispatch to handler
                self.dispatch_message(message)
        except Exception as e:
            print(f"[Transport:{self.client_id}] Connection error from {addr}: {e}")
        finally:
            client_socket.close()

    def recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """
        Receive exactly n bytes from socket

        As TCP is a streaming protocol hence send() may not send all bytes at once, and recv() may not receive all bytes at once, this function ensures we received exactly n bytes.

        Args: 
            sock: Socket to received from
            n: Number of bytes to receive
        
        Returns:
            Exactly n bytes, or empty bytes if connection closed
        """
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return b""  # Connection closed
            data += chunk
        return data
    
    def dispatch_message(self, message: Message):
        """Dispatch message to appropriate handler.

        Lock is held only during the dict lookup (microseconds), NOT during
        handler execution. This prevents relay broadcasts from blocking
        all incoming message reception.
        """
        with self.handler_lock:
            handler = self.message_handlers.get(message.message_type)

        if handler is not None:
            try:
                handler(message)
            except Exception as e:
                with self.stats_lock:
                    self.stats["errors"] += 1
                print(f"[Transport:{self.client_id}] Handler error: {e}")
        else:
            print(f"[Transport:{self.client_id}] No handler for '{message.message_type}'")

    def register_handler(self, message_type: str, handler: Callable):
        """
        Register handler for specific message type

        Args:
            message_type: Type of message to handle
            handler: Calback function(message: Message) -> None
        """
        with self.handler_lock:
            self.message_handlers[message_type] = handler
            print(f"[Transport:{self.client_id}] Registered handler for '{message_type}'")
    
    def register_peer(self, peer_id: str, host: str, port: int):
        """
        Register a peer's address for communication

        Args:
            peer_id: Peer's client ID
            host: Peer's hostname or IP address
            port: Peer's port nmber
        """
        self.peer_addresses[peer_id] = (host, port)
        print(f"[Transport:{self.client_id}]Registered peer {peer_id} at {host}:{port}")
    
    def get_or_create_connection(self, peer_id: str) -> Optional[socket.socket]:
        """
        Get existing connection or create new one to peer

        Args:
            peer_id: Peer to connect to
        
        Returns:
            Socket connection, or None if failed
        """
        # Check if we already have a connection
        with self.peer_lock:
            if peer_id in self.peer_sockets:
                sock = self.peer_sockets[peer_id]
                # Check if still alive
                try:
                    # Try to send 0 bytes to check connection
                    sock.send(b"", socket.MSG_DONTWAIT)
                    return sock
                except:
                    # Connection dead, remove it
                    del self.peer_sockets[peer_id]
    
        # Need to create new connection
        if peer_id not in self.peer_addresses:
            print(f"[Transport:{self.client_id}]No address for peer {peer_id}")
            return None
        
        host, port = self.peer_addresses[peer_id]
        
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)   # 5 second connection timeout
            
            # Connect to peer
            sock.connect((host, port))
            
            print(f"[Transport:{self.client_id}] Connected to {peer_id} at {host}:{port}")

            # Store connection
            with self.peer_lock:
                self.peer_sockets[peer_id] = sock
            return sock

        except Exception as e:
            print(f"[Transport:{self.client_id}] Failed to connect to {peer_id}: {e}")
            with self.stats_lock:
                self.stats["connections_failed"] += 1
            return None
    
    def send(self, receiver_id: str, message_type: str, payload: Any) -> bool:
        """
        Send message to peer

        Args:
            receiver_id: ID of the peer to send to
            message_type: Type of message
            payload: Message payload
        
        Returns:
            True if sent successfully, False otherwise
        """
        # Create message
        message = Message(
            sender_id = self.client_id,
            receiver_id = receiver_id,
            message_type = message_type,
            payload = payload,
            timestamp = time.time(),
            message_id=f"{self.client_id}_{time.time()}"
        )

        # Serialize
        message_data = message.to_bytes()
        message_length = len(message_data)

        # Get connection
        sock = self.get_or_create_connection(receiver_id)
        if sock is None:
            return False

        # Get or create per-peer send lock (prevents concurrent writes
        # from interleaving length+data on the same socket)
        with self.peer_lock:
            if receiver_id not in self.send_locks:
                self.send_locks[receiver_id] = threading.Lock()
            send_lock = self.send_locks[receiver_id]

        with send_lock:
            try:
                # Send length (4 bytes, network bytes order)
                length_bytes = struct.pack("!I", message_length)
                sock.sendall(length_bytes)

                # Send message data
                sock.sendall(message_data)

                # Update statistics
                with self.stats_lock:
                    self.stats["sent"] += 1
                    self.stats["bytes_sent"] += message_length

                print(f"[Transport:{self.client_id}] Sent '{message_type}' "
                      f"to {receiver_id} ({message_length} bytes)")

                return True

            except Exception as e:
                print(f"[Transport:{self.client_id}] Send error to {receiver_id}: {e}")

                # Remove dead connection
                with self.peer_lock:
                    if receiver_id in self.peer_sockets:
                        del self.peer_sockets[receiver_id]

                with self.stats_lock:
                    self.stats["errors"] += 1

                return False
        
    def broadcast(self, peer_ids: List[str], message_type: str, payload: Any) -> int:
        """
        Broadcst message to multiple peers

        Args:
            peer_ids: List of peer IDs
            message_type: Type of message
            payload: Message payload

        Returns:
            Number of successful sends
        """
        success_count = 0
        for peer_id in peer_ids:
            if self.send(peer_id, message_type, payload):
                success_count += 1
        
        print(f"[Transport:{self.client_id}] Broadcast to {success_count}/{len(peer_ids)} peers")
        return success_count
    
    def get_address(self) -> Tuple[str, int]:
        """
        Get this transport's listening address

        Returns:
            (host, port) tuple
        """
        return (self.host, self.port)

    def get_statistics(self) -> Dict:
        """Get transport statistics"""
        with self.stats_lock:
            return dict(self.stats)

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"TRANSPORT STATISTICS ({self.client_id})")
        print(f"{'='*60}")
        print(f"Sent: {stats['sent']} messages ({stats['bytes_sent']/1024:.1f} KB)")
        print(f"Received: {stats['received']} messages ({stats['bytes_received']/1024:.1f} KB)")
        print(f"Connections: {stats['connections_accepted']} accepted, "
              f"{stats['connections_failed']} failed")
        print(f"Errors: {stats['errors']}")
        print(f"Active Connections: {len(self.peer_sockets)}")
        print(f"{'='*60}\n")

    def shutdown(self):
        """Shutdown transport (close all connections)"""
        print(f"[Transport:{self.client_id}] Shutting down...")
        self.running = False

        # Close all peer connections
        with self.peer_lock:
            for sock in self.peer_sockets.values():
                try:
                    sock.close()
                except:
                    pass
            self.peer_sockets.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Wait for server thread
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout = 2.0)
        
        print(f"[Transport:{self.client_id}] Shutdown complete")

    def __del__(self):
        """Cleanup on destruction"""
        self.shutdown()

# Helper function for discovery/registration
def create_peer_registry() -> Dict[str, Tuple[str, int]]:
    """
    Create a simple peer registry (dictionary)
    """
    return {}


