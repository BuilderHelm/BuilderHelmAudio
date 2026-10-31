# BuilderHelm Audio demo

A small iPhone and iPad app for trying BuilderHelmAudio on a device: download a model, transcribe the
bundled samples or the microphone, spot keywords and benchmark speed.

1. Open this folder in Xcode.
2. Choose your device, and under **Signing & Capabilities** pick your team.
3. Run. On the device, enable **Settings → Privacy & Security → Developer Mode** if asked.

**Benchmark** transcribes a sample ten times after one warm-up and shows the median time, real-time
factor, device and whether the text matches the Mac reference in `Samples/mac-reference.json`.
The iOS Simulator has no Neural Engine, so its speed is not representative.

From the command line, launch with `--autobenchmark` to benchmark every model and sample and print the
results to the console:

```bash
xcrun devicectl device process launch --console --device <device> ai.builderhelm.audio.demo --autobenchmark
```

The samples are synthetic speech generated for this project.
