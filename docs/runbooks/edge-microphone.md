# Setting up an edge microphone (Raspberry Pi + XVF3800)

This document is a procedure. Follow it from a bare Raspberry Pi to a paired,
streaming edge microphone. Every value below is invented for illustration.
Replace `<host>`, `<token>`, and `<ingress host>` with your own values. Do not
put a real hostname, address, or credential into this file or into a copy of
it.

## 1. Hardware

1. Use a Raspberry Pi 4 with a 3 A power supply.
2. Attach a heatsink to the Pi. The XVF3800's own USB firmware runs the CPU
   hot under sustained capture (issue #4).
3. Connect a Seeed reSpeaker XVF3800 4-Mic Array to the Pi over USB.
4. Connect a powered speaker to the XVF3800's own 3.5 mm output. Do not
   connect the speaker to the Pi's own audio output.
5. The reply must leave through the array's own output, not the Pi's. Only
   the array's own chip holds the echo reference its AEC needs (D-14).

## 2. OS packages

1. Install the two native libraries this project's Python dependencies link
   against:

   ```bash
   sudo apt install -y libportaudio2 libusb-1.0-0
   ```

## 3. Install uv and clone the repository

1. Install `uv` system-wide, so that `sudo` can find it. The default
   installer puts `uv` in your own home directory, which is not on root's
   `PATH`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
   ```

2. Clone this repository into `/opt/atlas-edge`:

   ```bash
   sudo git clone https://github.com/<your-fork-or-this-repository>/atlas.git /opt/atlas-edge
   ```

3. Install the edge package's dependencies:

   ```bash
   cd /opt/atlas-edge/edge
   sudo env UV_PYTHON_INSTALL_DIR=/opt/atlas-edge/.python uv sync --frozen
   ```

4. The edge package needs Python 3.11. Raspberry Pi OS ships a newer
   Python, so `uv` downloads 3.11. `UV_PYTHON_INSTALL_DIR` puts that
   download under `/opt/atlas-edge`. Without it, the interpreter lands in
   root's home directory, and the `atlas-edge` service user cannot run it.

## 4. Create the service user

1. Create a dedicated, unprivileged system user for the service:

   ```bash
   sudo useradd --system --no-create-home --shell /usr/sbin/nologin atlas-edge
   ```

2. The systemd unit in step 8 also adds this user to the `audio` and
   `plugdev` groups through `SupplementaryGroups`. Do not add them by hand.

## 5. Install the udev rule

1. Copy the udev rule into place:

   ```bash
   sudo cp /opt/atlas-edge/edge/udev/99-atlas-edge-xvf3800.rules /etc/udev/rules.d/
   ```

2. Reload the rules:

   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

3. This rule grants USB access to the `plugdev` group, scoped to the
   XVF3800's own vendor and product ID (`2886:001a`) only.

## 6. Fetch the Silero VAD model

1. Run the fetch script once:

   ```bash
   cd /opt/atlas-edge/edge
   sudo scripts/fetch-vad-model.sh
   ```

2. This script checks the downloaded file against a pinned SHA-256 digest.
   A mismatch deletes the file and stops with an error. The service itself
   never downloads a model.

## 7. Fix the array's own output level

1. The XVF3800 ships with both of its own `PCM` mixer controls set low
   (`-23 dB` and `-20 dB`, about `-43 dB` combined). A reply at this level is
   close to inaudible.
2. Before the first reply, turn the powered speaker down. The next step
   raises the array's own output level by a large amount.
3. Set both controls to full and unmute them:

   ```bash
   amixer -c Array sset "PCM",0 100% unmute
   amixer -c Array sset "PCM",1 100% unmute
   sudo alsactl store Array
   ```

4. Turn the powered speaker back up to a comfortable level.

## 8. Free the array from PipeWire

1. This service opens the array by its own name, not by the Pi's default
   audio device. If a desktop PipeWire session already holds the array open
   under another user, this service cannot open it.
2. If a desktop session runs on this Pi, stop and disable PipeWire for that
   user before you start the service:

   ```bash
   systemctl --user stop pipewire pipewire-pulse
   systemctl --user disable pipewire pipewire-pulse
   ```

## 9. Pair the device from the admin webapp

1. Sign in to the admin webapp as an admin.
2. Open the **Edge devices** screen.
3. Under **Add device**, type a name for this Pi and click **Add device**.
4. Copy the token the screen shows. The screen shows this token only once
   (D-03).

## 10. Write the config file

1. Create the config directory:

   ```bash
   sudo mkdir -p /etc/atlas-edge
   ```

2. Write `/etc/atlas-edge/config.toml` with the server URL and the token
   from step 9:

   ```toml
   server_url = "wss://<host>/ws/edge"
   token = "<token>"
   ```

3. Set the file's owner and permissions. The service refuses to start if
   another user or group can read this file (D-03):

   ```bash
   sudo chown atlas-edge:atlas-edge /etc/atlas-edge/config.toml
   sudo chmod 600 /etc/atlas-edge/config.toml
   ```

## 11. Install and start the service

1. Copy the systemd unit into place:

   ```bash
   sudo cp /opt/atlas-edge/edge/systemd/atlas-edge.service /etc/systemd/system/
   ```

2. Reload systemd, then enable and start the service:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now atlas-edge
   ```

3. This unit restarts the service forever on any exit, with no limit on
   the number of restarts (D-04).

## 12. Select the edge microphone in the setup wizard

1. In the admin webapp, open the setup wizard's audio source step.
2. Select **Edge microphone (Raspberry Pi)**.
3. Restart the server. The audio source change takes effect only after a
   restart, and this turns the camera microphone off (D-15).

## 13. Check reachability

The Pi always dials out to the server. The server never needs the Pi's own
address (D-02).

**Under Helm:** the Pi reaches the server at `wss://<ingress host>/ws/edge`
through the ingress controller's own TLS. Use this same host in step 10's
`server_url`.

**Under Docker Compose:** the default port binding reaches the loopback
address only. A Pi on the LAN cannot reach this by default. Do one of the
following:

1. Widen the published port and put a TLS-terminating proxy in front of it.
   See [`deploy-compose.md`](deploy-compose.md) for the exact trade-off this
   makes.
2. Or, only on a LAN you trust, set `allow_plaintext = true` in the Pi's
   own `config.toml` and use a `ws://` URL. This sends the device token
   over the network with no encryption. State this risk to anyone who
   asks why this option exists.

## 14. Confirm it works

1. Watch the Pi's own service log:

   ```bash
   journalctl -u atlas-edge -f
   ```

2. On the **Edge devices** screen, confirm this device shows a **Connected**
   badge.
3. Speak a turn to the Pi. Open that turn's session in the admin webapp.
   Confirm `events.jsonl` holds an `edge.latency` event with an
   `added_delay_ms_p95` value.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The Pi's log shows an HTTP 403 on connect | The token is wrong, or an admin revoked it | Pair the device again (step 9) and rewrite `config.toml` (step 10) |
| The connection closes with code 4009 | Another Pi already holds this token's connection | Confirm only one Pi runs with this token, or pair a second device |
| The log shows "PortAudio library not found" | `libportaudio2` is not installed | Run step 2 again |
