#!/bin/bash
# Скрипт для добавления конфигурации Тоғызқұмалақ в существующий nginx конфиг

if [ -z "$1" ]; then
    echo "Использование: $0 <путь-к-nginx-конфигу>"
    echo ""
    echo "Примеры:"
    echo "  $0 /etc/nginx/sites-enabled/default"
    echo "  $0 /etc/nginx/sites-enabled/mysite"
    echo ""
    echo "Сначала найдите ваш активный nginx конфиг:"
    echo "  ls -la /etc/nginx/sites-enabled/"
    exit 1
fi

NGINX_CONFIG="$1"
LOCATIONS_FILE="/etc/nginx/sites-available/togyzqumalaq-locations.conf"

if [ ! -f "$NGINX_CONFIG" ]; then
    echo "❌ Файл не найден: $NGINX_CONFIG"
    exit 1
fi

if [ ! -f "$LOCATIONS_FILE" ]; then
    echo "❌ Файл с location блоками не найден: $LOCATIONS_FILE"
    echo "Сначала запустите setup_server.sh"
    exit 1
fi

# Проверить, не добавлена ли уже конфигурация
if grep -q "location /togyzqumalaq" "$NGINX_CONFIG"; then
    echo "⚠️  Конфигурация для /togyzqumalaq уже присутствует в $NGINX_CONFIG"
    read -p "Перезаписать? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    # Удалить старую конфигурацию
    sed -i '/location \/togyzqumalaq/,/^[[:space:]]*}/d' "$NGINX_CONFIG"
fi

# Найти server блок и добавить конфигурацию перед закрывающей скобкой
if grep -q "server {" "$NGINX_CONFIG"; then
    # Добавить перед последней закрывающей скобкой server блока
    sed -i '/^[[:space:]]*}[[:space:]]*$/{
        r '"$LOCATIONS_FILE"'
    }' "$NGINX_CONFIG"
    
    echo "✅ Конфигурация добавлена в $NGINX_CONFIG"
    echo ""
    echo "Проверка конфигурации nginx..."
    if nginx -t; then
        echo "✅ Конфигурация корректна"
        read -p "Перезагрузить nginx? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            systemctl reload nginx
            echo "✅ Nginx перезагружен"
        fi
    else
        echo "❌ Ошибка в конфигурации nginx!"
        echo "Проверьте файл: $NGINX_CONFIG"
        exit 1
    fi
else
    echo "❌ Server блок не найден в $NGINX_CONFIG"
    echo "Добавьте вручную содержимое из $LOCATIONS_FILE"
    exit 1
fi

