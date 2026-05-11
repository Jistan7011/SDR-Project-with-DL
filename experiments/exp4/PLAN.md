# Experiment 4 Plan: End-to-End OTA Classification + Payload Recovery

## Goal

Experiment 4 extends the previous IQ-window modulation classifier into an end-to-end receive chain:

```text
captured IQ -> modulation classifier -> DSP demodulator selection -> frame sync -> payload recovery -> CRC/BER/CER/packet metrics
```

The RF condition stays aligned with Experiment 3: authorized self-transmitted HackRF One to RTL-SDR Blog V4, 1 m OTA antenna path, face-to-face layout, no RF cable between SDRs.

## Scope

- Use the Experiment 3 best ResNet1D `[I,Q,instantaneous_frequency]` checkpoint as the default classifier.
- Capture BASK/BFSK/BPSK with the shared payload pool: `A, F, P, 0, 1, 7, K, R, S, Z`.
- Preserve both classifier input windows and longer DSP recovery IQ bursts.
- Report predicted demod recovery and oracle demod recovery separately so that classifier errors and DSP/frame errors are not mixed together.
- Use a larger Experiment 4 dataset than Experiment 3: 15 OTA sessions, session-held-out split, and 20 frame windows per capture.

## Dataset Scale

Default full run:

```text
sessions: 15
captures per session: 3 modulations x 10 payloads = 30 captures
frame windows per capture: 20
total frame-level samples: 15 x 30 x 20 = 9000

train: session_001 ~ session_009 = 5400 samples
val:   session_010 ~ session_012 = 1800 samples
test:  session_013 ~ session_015 = 1800 samples
```

This is intentionally larger than the Experiment 3 1 m OTA target-domain set. Experiment 4 is allowed to take longer because payload recovery metrics are more sensitive than modulation accuracy alone.

## Frame Format

```text
preamble 16 bits: 1010101010101010
sync word 8 bits: 11001100
payload 8 bits: ASCII 1-byte payload
CRC8 8 bits: polynomial 0x07, init 0x00
total: 40 bits
```

## Success Criteria

Minimum:

- classification accuracy >= 0.70
- oracle-demod CRC pass rate >= 0.60
- predicted-modulation payload recovery accuracy >= 0.40
- no modulation has CRC pass rate collapsed to 0

Target:

- classification accuracy >= 0.75
- oracle-demod CRC pass rate >= 0.75
- predicted-modulation payload recovery accuracy >= 0.60

## Follow-Up Experiments

- Experiment 5: controlled AWGN/SNR stress test
- Experiment 6: adjacent-channel interference and multi-signal separation
- Experiment 7: unknown/open-set RF identification
- Experiment 8: streaming real-time analysis pipeline
