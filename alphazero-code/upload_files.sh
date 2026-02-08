#!/bin/bash
# Скрипт для копирования файлов на сервер

SERVER="root@91.186.197.89"
SERVER_PASS="sP+FkvHKi-7,W2"
SERVER_DIR="/var/www/togyzqumalaq"

# Проверка наличия sshpass
if command -v sshpass &> /dev/null; then
    SSH_CMD="sshpass -p '$SERVER_PASS' ssh"
    SCP_CMD="sshpass -p '$SERVER_PASS' scp"
else
    echo "⚠️  sshpass не установлен. Будет запрошен пароль."
    SSH_CMD="ssh"
    SCP_CMD="scp"
fi

echo "📦 Копирование файлов на сервер..."

# Создать директории на сервере
$SSH_CMD $SERVER "mkdir -p $SERVER_DIR/static $SERVER_DIR/game_logs"

# Копировать статические файлы
echo "📄 Копирование статических файлов..."
$SCP_CMD index.html styles.css game.js mcts-worker.js $SERVER:$SERVER_DIR/static/

# Копировать серверные файлы
echo "🐍 Копирование серверных файлов..."
$SCP_CMD server.py requirements.txt setup_server.sh add_nginx_config.sh $SERVER:$SERVER_DIR/

echo "✅ Файлы скопированы!"
echo ""
echo "Теперь на сервере выполните:"
echo "  ssh $SERVER"
echo "  cd $SERVER_DIR"
echo "  chmod +x setup_server.sh"
echo "  ./setup_server.sh"

