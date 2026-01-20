#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import time
import queue
import threading
import argparse
import struct
from typing import Optional

# CRC-16 CCITT

def crc16_ccitt(data: bytes, poly=0x1021, init=0xFFFF):
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc

# Configuration

class MFSKConfig:
    SAMPLE_RATE = 48000
    CHUNK_SIZE = 4096

    SYMBOL_DURATION = 0.064      # 3072 samples (FFT friendly)
    GUARD_DURATION = 0.010

    BASE_FREQ = 2000
    FREQ_SPACING = 100

    DATA_TONES = 16
    START_SYMBOL = 16
    END_SYMBOL = 17
    NUM_TONES = 18

    PREAMBLE = [0, 15] * 4

    ENERGY_FLOOR = 1e-4
    DOM_RATIO_THRESHOLD = 1.3

    FRAME_TIMEOUT = 2.5

    def __init__(self):
        self.SAMPLES_PER_SYMBOL = int(self.SAMPLE_RATE * self.SYMBOL_DURATION)
        self.GUARD_SAMPLES = int(self.SAMPLE_RATE * self.GUARD_DURATION)
        self.SYMBOL_BLOCK = self.SAMPLES_PER_SYMBOL + self.GUARD_SAMPLES
        self.SLIDE_STEP = self.GUARD_SAMPLES // 2

        self.TONES = [
            self.BASE_FREQ + i * self.FREQ_SPACING
            for i in range(self.NUM_TONES)
        ]

# Transmitter

class MFSKTransmitter:
    def __init__(self, cfg: MFSKConfig):
        self.cfg = cfg

    def generate_symbol(self, freq):
        t = np.arange(self.cfg.SAMPLES_PER_SYMBOL) / self.cfg.SAMPLE_RATE
        tone = np.sin(2 * np.pi * freq * t)
        tone *= np.hanning(len(tone))
        guard = np.zeros(self.cfg.GUARD_SAMPLES)
        return np.concatenate([tone, guard]).astype(np.float32)

    def encode(self, data: bytes):
        payload = data + struct.pack(">H", crc16_ccitt(data))
        syms = self.cfg.PREAMBLE + [self.cfg.START_SYMBOL]

        for b in payload:
            syms.append((b >> 4) & 0xF)
            syms.append(b & 0xF)

        syms.append(self.cfg.END_SYMBOL)
        return syms

    def build_audio(self, data: bytes):
        symbols = self.encode(data)
        return np.concatenate([
            self.generate_symbol(self.cfg.TONES[s]) for s in symbols
        ])

    def send(self, text: str):
        audio = self.build_audio(text.encode())
        print(f"TX → {len(text)} bytes")
        sd.play(audio, self.cfg.SAMPLE_RATE)
        sd.wait()

# Receiver

class MFSKReceiver:
    def __init__(self, cfg: MFSKConfig):
        self.cfg = cfg
        self.running = False

        self.audio_queue = queue.Queue(maxsize=50)
        self.buffer = np.zeros(0, dtype=np.float32)

        self.symbols = []
        self.last_symbol_time = time.time()

        self.goertzel_coeffs = [
            2 * np.cos(2 * np.pi * f / self.cfg.SAMPLE_RATE)
            for f in self.cfg.TONES
        ]

    # Audio callback
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    # Goertzel detector
    
    def goertzel_energy(self, samples, coeff):
        s0 = s1 = s2 = 0.0
        for x in samples:
            s0 = x + coeff * s1 - s2
            s2 = s1
            s1 = s0
        return s1 * s1 + s2 * s2 - coeff * s1 * s2

    def detect_symbol(self, samples) -> Optional[int]:
        peak = np.max(np.abs(samples))
        if peak < self.cfg.ENERGY_FLOOR:
            return None

        samples = samples / (peak + 1e-9)

        energies = np.array([
            self.goertzel_energy(samples, c)
            for c in self.goertzel_coeffs
        ])

        best = np.argmax(energies)
        strongest = energies[best]
        second = np.partition(energies, -2)[-2]

        if strongest / (second + 1e-9) > self.cfg.DOM_RATIO_THRESHOLD:
            return int(best)

        return None

    # Frame decoder
    
    def try_decode(self) -> Optional[bytes]:
        p = self.cfg.PREAMBLE

        for i in range(len(self.symbols) - len(p)):
            if self.symbols[i:i+len(p)] != p:
                continue

            start = i + len(p)
            if start >= len(self.symbols):
                return None
            if self.symbols[start] != self.cfg.START_SYMBOL:
                continue

            data_syms = []
            for s in self.symbols[start+1:]:
                if s == self.cfg.END_SYMBOL:
                    break
                data_syms.append(s)

            if len(data_syms) < 4 or len(data_syms) % 2:
                return None

            raw = bytearray()
            for j in range(0, len(data_syms), 2):
                raw.append((data_syms[j] << 4) | data_syms[j+1])

            payload = bytes(raw)
            data, crc_rx = payload[:-2], payload[-2:]

            if crc16_ccitt(data) == struct.unpack(">H", crc_rx)[0]:
                self.symbols.clear()
                return data

        return None

    # DSP loop (timing-robust)
    
    def dsp_loop(self):
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                self.buffer = np.concatenate([self.buffer, chunk])

                while len(self.buffer) >= self.cfg.SAMPLES_PER_SYMBOL:
                    detected = False

                    for offset in range(
                        0,
                        len(self.buffer) - self.cfg.SAMPLES_PER_SYMBOL,
                        self.cfg.SLIDE_STEP
                    ):
                        sym = self.detect_symbol(
                            self.buffer[offset:offset + self.cfg.SAMPLES_PER_SYMBOL]
                        )
                        if sym is not None:
                            self.symbols.append(sym)
                            self.last_symbol_time = time.time()
                            self.buffer = self.buffer[
                                offset + self.cfg.SYMBOL_BLOCK:
                            ]
                            detected = True

                            decoded = self.try_decode()
                            if decoded:
                                print("RX ←", decoded.decode(errors="ignore"))
                            break

                    if not detected:
                        break

                if time.time() - self.last_symbol_time > self.cfg.FRAME_TIMEOUT:
                    self.symbols.clear()

            except queue.Empty:
                pass

    # Public API
    
    def listen(self, duration):
        print(f"RX listening for {duration:.1f} s")
        self.running = True

        dsp = threading.Thread(target=self.dsp_loop, daemon=True)
        dsp.start()

        with sd.InputStream(
            samplerate=self.cfg.SAMPLE_RATE,
            blocksize=self.cfg.CHUNK_SIZE,
            channels=1,
            callback=self.audio_callback,
            dtype="float32"
        ):
            time.sleep(duration)

        self.running = False
        dsp.join()

# Self-test

def self_test(cfg, message, duration):
    tx = MFSKTransmitter(cfg)
    rx = MFSKReceiver(cfg)

    def tx_loop():
        while rx.running:
            tx.send(message)
            time.sleep(1.0)

    rx.running = True
    threading.Thread(target=rx.dsp_loop, daemon=True).start()
    threading.Thread(target=tx_loop, daemon=True).start()

    with sd.InputStream(
        samplerate=cfg.SAMPLE_RATE,
        blocksize=cfg.CHUNK_SIZE,
        channels=1,
        callback=rx.audio_callback,
        dtype="float32"
    ):
        time.sleep(duration)

    rx.running = False

# CLI

def list_devices():
    print("\nAvailable audio devices:")
    for i, d in enumerate(sd.query_devices()):
        print(f" {i}: {d['name']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tx", "rx", "selftest"])
    parser.add_argument("--message", default="Hello MFSK")
    parser.add_argument("--duration", type=float, default=30)

    parser.add_argument("--device", type=int)
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)

    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.mode is None:
        parser.error("--mode is required unless --list-devices is used")

    if args.device is not None:
        sd.default.device = (args.device, args.device)
    else:
        if args.input_device is not None:
            sd.default.input_device = args.input_device
        if args.output_device is not None:
            sd.default.output_device = args.output_device

    cfg = MFSKConfig()

    if args.mode == "tx":
        MFSKTransmitter(cfg).send(args.message)
    elif args.mode == "rx":
        MFSKReceiver(cfg).listen(args.duration)
    else:
        self_test(cfg, args.message, args.duration)

if __name__ == "__main__":
    main()
