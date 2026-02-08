#!/bin/bash
# Скрипт для выполнения на сервере после копирования файлов

set -e

SERVER_DIR="/var/www/togyzqumalaq"
SERVICE_NAME="togyzqumalaq-logger"

echo "🚀 Настройка Тоғызқұмалақ на сервере..."

# 1. Создать директории
echo "📁 Создание директорий..."
mkdir -p $SERVER_DIR/{static,game_logs}
chmod 755 $SERVER_DIR/game_logs

# 2. Установить зависимости Python
echo "📥 Установка зависимостей Python..."
cd $SERVER_DIR
python3 -m pip install --user -r requirements.txt || pip3 install -r requirements.txt

# 3. Создать systemd service
echo "⚙️  Создание systemd service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=TogyzQumalaq Game Logger Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SERVER_DIR
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 $SERVER_DIR/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 4. Создать конфигурацию nginx для подпути
echo "🌐 Создание конфигурации nginx..."
cat > /etc/nginx/sites-available/togyzqumalaq << 'EOF'
# Конфигурация для Тоғызқұмалақ на подпути /togyzqumalaq
# Добавьте эти location блоки в существующий server блок вашего сайта

    # Статические файлы игры
    location /togyzqumalaq/ {
        alias /var/www/togyzqumalaq/static/;
        index index.html;
        try_files $uri $uri/ /togyzqumalaq/index.html;
    }

    # API для логирования
    location /togyzqumalaq/api {
        rewrite ^/togyzqumalaq/api(.*) $1 break;
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
EOF

# Создать файл с инструкциями для добавления в существующий конфиг
cat > /etc/nginx/sites-available/togyzqumalaq-locations.conf << 'EOF'
    # Статические файлы игры Тоғызқұмалақ
    location /togyzqumalaq/ {
        alias /var/www/togyzqumalaq/static/;
        index index.html;
        try_files $uri $uri/ /togyzqumalaq/index.html;
    }

    # API для логирования Тоғызқұмалақ
    location /togyzqumalaq/api/ {
        proxy_pass http://127.0.0.1:5000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
EOF

# 5. Инструкции по добавлению в nginx
echo ""
echo "⚠️  ВАЖНО: Нужно добавить конфигурацию в существующий server блок nginx!"
echo ""
echo "Содержимое для добавления сохранено в:"
echo "  /etc/nginx/sites-available/togyzqumalaq-locations.conf"
echo ""
echo "Добавьте содержимое этого файла в ваш существующий server блок"
echo "в файле /etc/nginx/sites-enabled/ваш-сайт"
echo ""
echo "Или выполните:"
echo "  cat /etc/nginx/sites-available/togyzqumalaq-locations.conf >> /etc/nginx/sites-enabled/ваш-сайт"
echo ""
read -p "Нажмите Enter после добавления конфигурации в nginx, чтобы проверить и перезагрузить..."

# Проверка конфигурации nginx
if nginx -t; then
    systemctl reload nginx
    echo "✅ Nginx перезагружен"
else
    echo "❌ Ошибка в конфигурации nginx!"
    echo "Проверьте конфигурацию: nginx -t"
    exit 1
fi

# 6. Запустить и включить Flask сервис
echo "🔄 Запуск сервиса логирования..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME

# 7. Проверить статус
echo "✅ Проверка статуса..."
sleep 2
systemctl status $SERVICE_NAME --no-pager -l | head -20

echo ""
echo "✨ Настройка завершена!"
echo ""
echo "⚠️  ВАЖНО: После добавления конфигурации в nginx игра будет доступна:"
echo "  🌐 http://91.186.197.89/togyzqumalaq/"
echo "  📊 http://91.186.197.89/togyzqumalaq/api/health"
echo ""
echo "Для добавления конфигурации используйте:"
echo "  ./add_nginx_config.sh /etc/nginx/sites-enabled/ваш-сайт"
echo ""
echo "Полезные команды:"
echo "  systemctl status $SERVICE_NAME  - статус сервиса"
echo "  systemctl restart $SERVICE_NAME  - перезапуск сервиса"
echo "  journalctl -u $SERVICE_NAME -f   - логи сервиса"

