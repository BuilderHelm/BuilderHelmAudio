#!/bin/bash
set -euo pipefail

# Use a simulator from the selected Xcode rather than depending on one phone's display name.
bhaudio_simulator_id=$(python3 - <<'PY'
import json
import re
import subprocess

devices = json.loads(subprocess.check_output(["xcrun", "simctl", "list", "devices", "available", "--json"]))["devices"]
sdk = subprocess.check_output(["xcrun", "--sdk", "iphonesimulator", "--show-sdk-version"], text=True).strip()
sdk_version = tuple(int(part) for part in sdk.split(".")[:2])
candidates = []
for runtime, phones in devices.items():
    version = re.search(r"\.iOS-(\d+)-(\d+)", runtime)
    if not version or int(version[1]) < 18 or (int(version[1]), int(version[2])) > sdk_version:
        continue
    for phone in phones:
        if phone.get("isAvailable") and phone["name"].startswith("iPhone"):
            candidates.append((phone["state"] == "Booted", int(version[1]), int(version[2]), phone["udid"]))
if not candidates:
    raise SystemExit("No available iPhone simulator with iOS 18+ for the selected Xcode")
print(max(candidates)[3])
PY
)

xcodebuild test \
  -scheme BuilderHelmAudio-Package \
  -destination "platform=iOS Simulator,id=$bhaudio_simulator_id" \
  -derivedDataPath .build/ci-ios \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
