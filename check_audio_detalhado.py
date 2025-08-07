import sounddevice as sd

print("--- Detalhes dos Dispositivos de Áudio ---")
try:
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # Vamos focar apenas nos dispositivos que têm canais de entrada (microfones)
        if device['max_input_channels'] > 0:
            print(f"\nDispositivo de Entrada #{i}: {device['name']}")
            print(f"  Canais de Entrada: {device['max_input_channels']}")
            print(f"  Taxa de Amostragem Padrão: {device['default_samplerate']} Hz")
except Exception as e:
    print(f"Ocorreu um erro: {e}")

print("\n--- Fim da Lista ---")