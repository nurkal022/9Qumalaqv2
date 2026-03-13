#!/usr/bin/env python3
"""Deploy Togyzkumalaq web demo to remote server."""
import paramiko
import os
import sys

HOST = '5.129.198.203'
USER = 'root'
PASS = 'j2EYHhU+eAheZm'
REMOTE_DIR = '/opt/togyzkumalaq'

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    ssh.connect(HOST, username=USER, password=PASS)
    print("Connected!")

    def run(cmd, show=True):
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if show and out.strip():
            print(out.strip())
        if err.strip():
            print(f"STDERR: {err.strip()}")
        return out.strip()

    # Check server
    run("uname -a")
    run("free -h | head -2")
    run(f"nproc")

    # Check/install deps
    run(f"mkdir -p {REMOTE_DIR}/engine {REMOTE_DIR}/web")

    # Check for Rust
    has_rust = run("which rustc 2>/dev/null", show=False)
    if not has_rust:
        print("Installing Rust...")
        run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        run("source ~/.cargo/env && rustc --version")

    # Check for Python/Flask
    has_flask = run("python3 -c 'import flask' 2>&1 && echo ok || echo no", show=False)
    if 'ok' not in has_flask:
        print("Installing Flask...")
        run("pip3 install flask --break-system-packages 2>/dev/null || pip3 install flask")

    # Upload files via SFTP
    sftp = ssh.open_sftp()

    local_base = os.path.dirname(os.path.abspath(__file__))

    files_to_upload = [
        ('web/index.html', f'{REMOTE_DIR}/web/index.html'),
        ('web/server.py', f'{REMOTE_DIR}/web/server.py'),
        ('engine/nnue_weights.bin', f'{REMOTE_DIR}/engine/nnue_weights.bin'),
        ('engine/egtb.bin', f'{REMOTE_DIR}/engine/egtb.bin'),
        ('engine/Cargo.toml', f'{REMOTE_DIR}/engine/Cargo.toml'),
    ]

    # Check if opening book exists
    book_path = os.path.join(local_base, 'web', 'opening_book.json')
    if os.path.exists(book_path):
        files_to_upload.append(('web/opening_book.json', f'{REMOTE_DIR}/web/opening_book.json'))

    # Upload engine opening book (text format)
    engine_book = os.path.join(local_base, 'engine', 'opening_book.txt')
    if os.path.exists(engine_book):
        files_to_upload.append(('engine/opening_book.txt', f'{REMOTE_DIR}/engine/opening_book.txt'))

    # Upload engine source
    engine_src = os.path.join(local_base, 'engine', 'src')
    run(f"mkdir -p {REMOTE_DIR}/engine/src")
    for f in os.listdir(engine_src):
        if f.endswith('.rs'):
            files_to_upload.append((f'engine/src/{f}', f'{REMOTE_DIR}/engine/src/{f}'))

    for local_rel, remote_path in files_to_upload:
        local_path = os.path.join(local_base, local_rel)
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_rel}")
            continue
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"  Uploading {local_rel} ({size_mb:.1f} MB)...")
        try:
            sftp.put(local_path, remote_path)
        except Exception as e:
            print(f"    Error: {e}")

    sftp.close()

    # Build engine on server
    print("\nBuilding engine on server...")
    build_out = run(f"source ~/.cargo/env 2>/dev/null; cd {REMOTE_DIR}/engine && cargo build --release 2>&1 | tail -5")

    # Stop any existing instance
    run("pkill -f 'python3.*server.py' 2>/dev/null || true", show=False)
    run("pkill -f 'togyzkumalaq-engine' 2>/dev/null || true", show=False)

    # Create systemd service
    service = f"""[Unit]
Description=Togyzkumalaq AI Web Server
After=network.target

[Service]
Type=simple
WorkingDir={REMOTE_DIR}/web
ExecStart=/usr/bin/python3 {REMOTE_DIR}/web/server.py
Restart=always
RestartSec=5
Environment=PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
"""

    stdin, stdout, stderr = ssh.exec_command(f"cat > /etc/systemd/system/togyzkumalaq.service << 'HEREDOC'\n{service}HEREDOC")
    stdout.read()

    # Fix server.py paths for remote
    run(f"""sed -i "s|ENGINE_DIR = .*|ENGINE_DIR = '{REMOTE_DIR}/engine'|" {REMOTE_DIR}/web/server.py""")

    run("systemctl daemon-reload")
    run("systemctl enable togyzkumalaq")
    run("systemctl restart togyzkumalaq")

    import time
    time.sleep(3)
    status = run("systemctl status togyzkumalaq | head -15")

    # Open firewall
    run("ufw allow 8080 2>/dev/null || iptables -I INPUT -p tcp --dport 8080 -j ACCEPT 2>/dev/null || true", show=False)

    print(f"\n{'='*50}")
    print(f"Deployed! Access at: http://{HOST}:8080")
    print(f"{'='*50}")

    ssh.close()

if __name__ == '__main__':
    main()
