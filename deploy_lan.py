#!/usr/bin/env python3
"""Deploy Togyzkumalaq engine + web to LAN server 10.0.34.22.

Strategy: upload pre-built binary + source files + restart service.
We DON'T build on the remote — we trust our local x86-64-v3 binary.
"""
import os
import sys
import time
import paramiko

HOST = '10.0.34.22'
USER = 'llama'
PASS = os.environ.get("DEPLOY_PASSWORD")
REMOTE_DIR = '/home/llama/togyz_server'

LOCAL = os.path.dirname(os.path.abspath(__file__))


def main():
    if PASS is None:
        raise RuntimeError("Set DEPLOY_PASSWORD env var")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    ssh.connect(HOST, username=USER, password=PASS, timeout=10)
    print("Connected.")

    def run(cmd, sudo=False, show=True):
        if sudo:
            cmd = f"echo '{PASS}' | sudo -S {cmd}"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if show and out.strip():
            print(out.strip())
        if err.strip() and 'sudo' not in err and '[sudo]' not in err:
            sys.stderr.write(f"  STDERR: {err.strip()}\n")
        return out.strip()

    print(f"\n--- Pre-deploy: stopping any running engine/server ---")
    # Try systemd first, then plain pkill
    run("sudo systemctl stop togyzkumalaq 2>/dev/null || true", sudo=True, show=False)
    run("pkill -f 'python3.*server.py' 2>/dev/null || true", show=False)
    run("pkill -f 'togyzkumalaq-engine' 2>/dev/null || true", show=False)
    time.sleep(2)

    # Free port if still held
    run(f"echo '{PASS}' | sudo -S fuser -k 8080/tcp 2>/dev/null || true", show=False)
    time.sleep(1)

    print(f"\n--- Ensuring directories exist ---")
    run(f"mkdir -p {REMOTE_DIR}/engine/target/release {REMOTE_DIR}/engine/src {REMOTE_DIR}/web {REMOTE_DIR}/web/games_log")

    sftp = ssh.open_sftp()

    def upload(local_rel, remote_path, mode=None):
        local_path = os.path.join(LOCAL, local_rel)
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_rel}")
            return False
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"  → {local_rel} ({size_mb:.2f} MB) → {remote_path}")
        # Use open+write so binaries-in-use can be replaced safely
        try:
            with open(local_path, 'rb') as src:
                with sftp.open(remote_path, 'wb') as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
            if mode is not None:
                sftp.chmod(remote_path, mode)
            return True
        except Exception as e:
            print(f"    Error: {e}")
            return False

    print(f"\n--- Uploading engine binary ---")
    upload('engine/target/release/togyzkumalaq-engine',
           f'{REMOTE_DIR}/engine/target/release/togyzkumalaq-engine',
           mode=0o755)

    print(f"\n--- Uploading engine assets ---")
    upload('engine/nnue_weights.bin', f'{REMOTE_DIR}/engine/nnue_weights.bin')
    upload('engine/egtb.bin', f'{REMOTE_DIR}/engine/egtb.bin')
    upload('engine/opening_book.txt', f'{REMOTE_DIR}/engine/opening_book.txt')

    print(f"\n--- Uploading source (for reference) ---")
    src_dir = os.path.join(LOCAL, 'engine', 'src')
    for f in sorted(os.listdir(src_dir)):
        if f.endswith('.rs'):
            upload(f'engine/src/{f}', f'{REMOTE_DIR}/engine/src/{f}')
    upload('engine/Cargo.toml', f'{REMOTE_DIR}/engine/Cargo.toml')

    print(f"\n--- Uploading web ---")
    upload('web/index.html', f'{REMOTE_DIR}/web/index.html')
    upload('web/server.py', f'{REMOTE_DIR}/web/server.py')
    if os.path.exists(os.path.join(LOCAL, 'web', 'opening_book.json')):
        upload('web/opening_book.json', f'{REMOTE_DIR}/web/opening_book.json')

    sftp.close()

    print(f"\n--- Patching server.py paths for remote ---")
    run(f"sed -i \"s|ENGINE_DIR = .*|ENGINE_DIR = '{REMOTE_DIR}/engine'|\" {REMOTE_DIR}/web/server.py")

    print(f"\n--- Verifying binary ---")
    out = run(f"ls -la {REMOTE_DIR}/engine/target/release/togyzkumalaq-engine")
    out = run(f"file {REMOTE_DIR}/engine/target/release/togyzkumalaq-engine 2>/dev/null || echo 'file cmd missing'")

    print(f"\n--- Quick smoke test ---")
    smoke = run(
        f"cd {REMOTE_DIR}/engine && timeout 8 ./target/release/togyzkumalaq-engine bench 2>&1 | tail -5 || echo 'bench-failed'"
    )

    print(f"\n--- Setting up systemd service ---")
    service = f"""[Unit]
Description=Togyzkumalaq AI Web Server
After=network.target

[Service]
Type=simple
User=llama
WorkingDirectory={REMOTE_DIR}/web
ExecStart=/usr/bin/python3 {REMOTE_DIR}/web/server.py
Restart=always
RestartSec=5
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=TT_SIZE_MB=2048

[Install]
WantedBy=multi-user.target
"""
    # Write service file via sudo tee
    run(f"echo '{PASS}' | sudo -S bash -c \"cat > /etc/systemd/system/togyzkumalaq.service << 'EOF'\n{service}EOF\"", show=False)
    run("sudo systemctl daemon-reload", sudo=True)
    run("sudo systemctl enable togyzkumalaq 2>&1", sudo=True)
    run("sudo systemctl restart togyzkumalaq", sudo=True)

    print(f"\n--- Waiting for startup ---")
    time.sleep(4)

    status = run("sudo systemctl status togyzkumalaq --no-pager 2>&1 | head -20", sudo=True)
    print(status)

    print(f"\n--- Checking port 8080 ---")
    run("ss -ltnp 2>/dev/null | grep 8080 || netstat -ltnp 2>/dev/null | grep 8080 || echo 'port not listening'")

    print(f"\n{'='*50}")
    print(f"Deployed! Access at: http://{HOST}:8080")
    print(f"{'='*50}")

    ssh.close()


if __name__ == '__main__':
    main()
