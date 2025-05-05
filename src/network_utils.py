import socket
import time

def create_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3.0)
    return sock

def is_socket_connected(sock):
    try:
        sock.getpeername()
        sock.send(b'')
        return True
    except (socket.error, OSError, AttributeError):
        return False

def safe_send(sock, command, max_retries=2):
    for attempt in range(max_retries):
        try:
            sock.sendall(command)
            return True
        except (socket.error, BrokenPipeError) as e:
            if attempt == max_retries - 1:
                print(f"! Send failed after {max_retries} attempts: {str(e)}")
                return False
            time.sleep(0.1 * (attempt + 1))
    return False
