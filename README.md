MFSK AUDIO COMMUNICATION SYSTEM

OVERVIEW

This program implements a simple and reliable audio-based data communication system using M-ary Frequency Shift Keying (MFSK). Text messages are transmitted and received over standard audio hardware such as speakers and microphones.

The system operates entirely in the audible frequency range and is just a fun project of mine. I didn'd use ultrasonic frequencies as most cheap microphones don't reliably receive them.

!! Currently only tested with blackhole2ch loopback !!

MAIN FEATURES

* Audible-band MFSK modulation
* Real-time audio input and output
* CRC-16 (CCITT) error detection
* Timing-tolerant symbol detection
* Self-test mode for easy validation
* Command-line interface

HIGH-LEVEL DESIGN

The program consists of four main parts:

1. Configuration (MFSKConfig)
   Holds all signal, timing, and protocol parameters.

2. Transmitter (MFSKTransmitter)
   Encodes data into MFSK symbols and generates audio.

3. Receiver (MFSKReceiver)
   Detects symbols from audio input and reconstructs messages.

4. Command-line Interface
   Allows selecting transmit, receive, or self-test modes.

SIGNAL PARAMETERS

Sample rate: 48000 Hz

Each symbol consists of:

* A sine wave tone
* A following guard interval of silence

Timing:

* Symbol duration: 64 ms
* Guard duration: 10 ms

Frequencies:

* Base frequency: 2000 Hz
* Frequency spacing: 100 Hz

Total tones: 18

* 16 data tones (values 0 to 15)
* 1 start-of-frame tone
* 1 end-of-frame tone

FRAME STRUCTURE

A transmitted frame has the following order:

1. Preamble
   A repeating symbol pattern used for synchronization.

2. Start symbol
   Marks the beginning of the payload.

3. Payload data
   Each byte is split into two 4-bit values (nibbles).
   Each nibble is transmitted as one MFSK symbol.

4. CRC
   A 16-bit CRC-16 CCITT checksum appended to the payload.

5. End symbol
   Marks the end of the frame.

TRANSMITTER OPERATION

The transmitter performs these steps:

* Convert text to bytes
* Compute and append CRC-16
* Split bytes into 4-bit nibbles
* Add preamble, start, and end symbols
* Generate a sine wave for each symbol frequency
* Apply a Hann window to reduce spectral leakage
* Append a guard interval after each symbol
* Play the resulting audio waveform

RECEIVER OPERATION

Audio Capture:

Audio samples are captured using a callback-based input stream and stored in a queue. A dedicated DSP thread processes the incoming audio asynchronously.

Symbol Detection:

For each candidate symbol window:

* The signal is normalized
* The Goertzel algorithm is applied for each tone frequency
* Energy is measured per frequency
* The strongest frequency is selected

A symbol is accepted only if:

* The signal energy exceeds a minimum threshold
* The strongest frequency is sufficiently stronger than the second strongest

This helps reject noise and ambiguous detections.

Timing Robustness:

The receiver does not assume perfect symbol alignment. Instead, it:

* Slides a detection window across the audio buffer
* Uses overlapping steps
* Removes detected symbols plus guard time from the buffer

This approach makes the decoder more tolerant to jitter and timing drift.

Frame Decoding:

Detected symbols are buffered and scanned for a valid frame:

* Preamble
* Start symbol
* Payload symbols
* End symbol

When a frame is found:

* Nibbles are reassembled into bytes
* CRC-16 is verified

If the CRC is valid, the decoded message is printed. If no valid frame is completed within a timeout, the buffer is cleared.

SELF-TEST MODE

Self-test mode runs the transmitter and receiver at the same time:

* The transmitter repeatedly sends a message
* The receiver listens on the microphone input

This mode is useful for:

* Verifying audio device configuration
* Testing symbol detection and decoding
* Debugging timing and threshold parameters

Typically, speaker output must be audible to the microphone for this to work.

COMMAND-LINE USAGE

List audio devices:

--list-devices

Transmit a message:

--mode tx --message "Hello"

Receive messages:

--mode rx --duration 30

Run self-test:

--mode selftest --message "Test" --duration 30

Audio device selection:

--device N
--input-device N
--output-device N

NOTES AND LIMITATIONS

* This system is designed for low data rates
* Audio quality and environment strongly affect reliability
* Transmissions are audible to humans
* Background noise and echo can reduce performance


