# Быстрое развертывание на сервере

## Способ 1: Автоматический (рекомендуется)

```bash
# 1. Загрузить файлы и настроить все автоматически
./deploy.sh
```

## Способ 2: Пошаговый

### Шаг 1: Загрузить файлы на сервер

```bash
./upload_files.sh
```

### Шаг 2: Подключиться к серверу и настроить

```bash
ssh root@91.186.197.89
# Пароль: sP+FkvHKi-7,W2

cd /var/www/togyzqumalaq
chmod +x setup_server.sh
./setup_server.sh
```

## Способ 3: Полностью вручную

См. подробную инструкцию в `DEPLOY.md`

## Важно: Добавить в существующий nginx конфиг

После выполнения `setup_server.sh` нужно добавить конфигурацию в ваш существующий nginx:

```bash
# На сервере
cd /var/www/togyzqumalaq
chmod +x add_nginx_config.sh

# Найти ваш nginx конфиг
ls -la /etc/nginx/sites-enabled/

# Добавить конфигурацию
./add_nginx_config.sh /etc/nginx/sites-enabled/ваш-сайт
```

Или вручную:
```bash
cat /etc/nginx/sites-available/togyzqumalaq-locations.conf >> /etc/nginx/sites-enabled/ваш-сайт
nginx -t && systemctl reload nginx
```

## Проверка работы

После развертывания проверьте:

1. **Игра доступна**: http://91.186.197.89/togyzqumalaq/
2. **API работает**: http://91.186.197.89/togyzqumalaq/api/health
3. **Сервис запущен**: 
   ```bash
   systemctl status togyzqumalaq-logger
   ```

## Обновление после изменений

```bash
# Загрузить обновленные файлы
./upload_files.sh

# На сервере перезапустить сервис
ssh root@91.186.197.89 "systemctl restart togyzqumalaq-logger"
```

## Логи и отладка

```bash
# Логи Flask сервера
ssh root@91.186.197.89 "journalctl -u togyzqumalaq-logger -f"

# Логи Nginx
ssh root@91.186.197.89 "tail -f /var/log/nginx/error.log"
```

