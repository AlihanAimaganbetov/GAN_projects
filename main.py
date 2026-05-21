import serial

PORT = "/dev/rfcomm0" # Или "COM3" для Windows
BAUD = 921600

try:
    # Открываем порт
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"--- Подключено к {PORT}. Нажмите Ctrl+C для выхода ---")

    while True:
        if ser.in_waiting > 0:
            # Читаем строку из порта
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)

except KeyboardInterrupt:
    print("\nОстановка...")
except Exception as e:
    print(f"Ошибка: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()