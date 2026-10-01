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

4. You can change the speaker level by voice, for example "turn it up" or
   "set the volume to 60". The Pi then changes one ALSA mixer control with
   `amixer`. To use a different card or control, add these keys to
   `config.toml`:

   | Key | Default | Meaning |
   |---|---|---|
   | `volume_card` | `Array` | The ALSA card name that `amixer -c` takes. |
   | `volume_control` | `PCM,0` | The mixer control. A value must not start with `-`. |

   The server keys `edge.volume.min_percent`, `edge.volume.max_percent`, and
   `edge.volume.step_percent` (30, 100, and 10) limit every level. Restart the
   edge service after you change a key on the Pi.

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

## 12. Turn on automatic updates

The Pi follows the server's build commit. Every 15 minutes, the Pi reads the
commit from the server's `/health` route. The first check is 5 minutes after
boot. If the Pi runs a different commit, it checks out that commit and
installs it the same way as steps 3 and 11. Then it restarts the edge
service. If the service does not stay up for 30 seconds, the Pi goes back to
the previous commit.

1. Copy the two updater units into place:

   ```bash
   sudo cp /opt/atlas-edge/edge/systemd/atlas-edge-update.service \
     /opt/atlas-edge/edge/systemd/atlas-edge-update.timer /etc/systemd/system/
   ```

2. Reload systemd, then enable and start the timer:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now atlas-edge-update.timer
   ```

3. Optional: run one check now:

   ```bash
   sudo systemctl start atlas-edge-update.service
   ```

4. Read the log of the updater:

   ```bash
   journalctl -u atlas-edge-update
   ```

Keep these facts in mind:

- The updater runs only commits that are on `main` of the repository you
  cloned in step 3. It refuses any other commit and logs the refusal. The
  server must build from that same repository.
- The updater refreshes only the unit files that are already in
  `/etc/systemd/system`. Optional units stay off until you install them.
  The `atlas-librespot` and `atlas-librespot-watchdog` units stay off until you install them in step 17.
- Build the server image with the `GIT_SHA` build argument set to the full
  commit. Without it, `/health` reports an empty commit and the updater
  does nothing:

  ```bash
  docker build --build-arg GIT_SHA="$(git rev-parse HEAD)" -t <your-registry>/atlas:<tag> .
  ```

- A restart during a spoken turn drops that turn.

To turn the updater off:

```bash
sudo systemctl disable --now atlas-edge-update.timer
```

## 13. Select the edge microphone in the setup wizard

1. In the admin webapp, open the setup wizard's audio source step.
2. Select **Edge microphone (Raspberry Pi)**.
3. Restart the server. The audio source change takes effect only after a
   restart, and this turns the camera microphone off (D-15).

## 14. Check reachability

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

## 15. Confirm it works

1. Watch the Pi's own service log:

   ```bash
   journalctl -u atlas-edge -f
   ```

2. On the **Edge devices** screen, confirm this device shows a **Connected**
   badge.
3. Speak a turn to the Pi. Open that turn's session in the admin webapp.
   Confirm `events.jsonl` holds an `edge.latency` event with an
   `added_delay_ms_p95` value.

## 16. Speaker identification (optional)

Speaker identification learns the voice of each household member. It can stop a
turn from a person who is not enrolled. It works on the edge microphone only.
A camera turn always records speaker "unknown".

Three modes exist: `off`, `record`, and `enforce`. The default is `off`. Skip
this section if you do not need this feature.

1. Fetch both speaker-id models into the running container:

   ```bash
   docker compose exec app env PYTHONPATH=src .venv/bin/python scripts/fetch_models.py --only speaker-id
   ```

2. Open your config file. Confirm the `speaker_id:` block sets `model`,
   `window_ms`, `speech_rms_floor`, and `change_similarity_floor`. A prior
   spike on this project measured these four values. Do not guess at them.
   The shipped config reads `model` from the `SPEAKER_ID_MODEL` environment
   variable. Set it to `campplus` or `titanet_small`. For Docker Compose, set
   it in `.env`. For Helm, set it in the runtime Secret.
3. Sign in to the admin webapp as an admin.
4. Open the **Speakers** screen.
5. Add each household member. For each member, read the five prompted
   phrases into the Pi's own microphone.
6. Set `SPEAKER_ID_MODE=record` in the same place as `SPEAKER_ID_MODEL`.
   On Compose, run `docker compose up -d app`. The command `docker compose
   restart` does not read `.env` again. On Helm, restart the pod after you
   change the Secret.
7. Speak several ordinary turns to the Pi as each enrolled member. Let some
   turns come from an unenrolled voice too, for example a visitor or the
   television.
8. Label those turns. Run this command once for each enrolled member, using
   that member's id from the Speakers screen:

   ```bash
   docker compose exec app env PYTHONPATH=src .venv/bin/python scripts/tune_speaker_threshold.py label \
       --sessions /data/sessions --labels /data/speaker-labels.json \
       --since <start time> --until <end time> --speaker-id <member id>
   ```

   Run the same command once more with `--other` in place of
   `--speaker-id <member id>`, for the turns from unenrolled voices.
9. Tune the threshold from the labeled turns:

   ```bash
   docker compose exec app env PYTHONPATH=src .venv/bin/python scripts/tune_speaker_threshold.py tune \
       --sessions /data/sessions --labels /data/speaker-labels.json
   ```

   Copy the printed `recommended_threshold` value into `speaker_id.threshold`
   in your config file.
10. Set `SPEAKER_ID_MODE=enforce` in the same place. Apply the change the
    same way as in step 6.
11. Measure the time speaker identification adds to a turn:

    ```bash
    docker compose exec app env PYTHONPATH=src .venv/bin/python scripts/measure_speaker_id.py \
        --sessions /data/sessions
    ```

    Confirm the report's `verdict` field reads `PASS`.

**Add a voice from a recorded turn.** You can add a voice that the assistant
missed. Use a turn that the Pi recorded before. The recording must still exist.
The `debug.retain_days` setting controls this. The default is 7 days.

1. Open the **Speakers** screen. Find **Recent unrecognized voices**. This list
   shows recent edge turns that no member matched.
2. Play the turn. Make sure that you hear only that member. Do not use a turn
   with the television or another person.
3. Select **Assign**. Select the member, or select **New member** and type a
   name. Then select **Assign voice**.

The `speaker_id.model` setting must be set. The mode can be `off`. A member
keeps at most 20 clips from recordings. The next clip removes the oldest one.
The five prompted phrases stay. The list **Clips from recordings** shows the
clips under the member. Each clip has the same weight as a prompted phrase. If
a clip is wrong, select **Remove**. The turn then returns to the inbox.

A recording of an enrolled member's voice can also pass this check. This is a
known limit, not a defect. The speaker label only personalizes the reply. It
never grants permission for anything.

**Limit home control for a member.** You can stop one member from changing
devices by voice.

1. Sign in to the admin webapp as an admin.
2. Open the **Speakers** screen.
3. Find the member. Clear **Can control home devices**.

The change applies to the next turn. You do not restart anything. The setting
works only when `SPEAKER_ID_MODE` is `enforce`. In `off` and `record` mode, no
check runs. In `enforce` mode, when the assistant identifies that member, it
refuses every request that changes a device. It also refuses a scheduled
workflow that has a device step. It speaks one fixed sentence, for example
"Alex, you can't control the house." The member can still ask questions, read
device states, and set reminders that have no device step. A voice match never
grants permission. It can only take home control away. A recording or a similar
voice of a permitted member can pass the check. The setting stops casual use.
It does not stop a determined attacker.

Both models come from the sherpa-onnx project's own release. Step 1 downloads
them. This repository does not store them. The CAM++ model carries the
Apache-2.0 license. The TitaNet-small model carries the CC-BY-4.0 license,
which requires attribution.

## 17. Play Spotify through the array (optional)

A request to play music, with no speaker named, plays through the 3.5 mm
jack of the array. The `librespot` program receives Spotify Connect and
writes audio into a FIFO. The edge service mixes that audio into its own
output stream. The music gets quiet while the assistant listens and while
it replies.

1. Install the `librespot` program from the `raspotify` apt package. Then
   stop and disable the `raspotify` service. Its unit opens ALSA directly
   and cannot share the array with the edge service:

   ```bash
   sudo systemctl disable --now raspotify
   ```

   Then create the system user that runs `librespot`:

   ```bash
   sudo useradd --system --no-create-home --shell /usr/sbin/nologin \
     --user-group atlas-librespot
   ```

2. To change the device name from the default `Atlas`, write this line in
   `/etc/atlas-edge/librespot.env`:

   ```
   LIBRESPOT_NAME=<name>
   ```

3. Copy the updated edge unit, the new `librespot` unit, and its watchdog
   unit into place. Restart the edge service first, because it creates the
   FIFO directory:

   ```bash
   sudo cp /opt/atlas-edge/edge/systemd/atlas-edge.service \
           /opt/atlas-edge/edge/systemd/atlas-librespot.service \
           /opt/atlas-edge/edge/systemd/atlas-librespot-watchdog.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl restart atlas-edge
   sudo systemctl enable --now atlas-librespot
   sudo systemctl enable --now atlas-librespot-watchdog
   ```

   The watchdog restarts `atlas-librespot` when its log shows "Connection
   to server closed". librespot 0.8.0 does not connect again after this
   error. If `atlas-librespot` already runs from an earlier install, copy
   only `atlas-librespot-watchdog.service`. Then run
   `sudo systemctl daemon-reload` and the last enable line.

4. Sign in once. On a phone on the house network, open Spotify and pick
   the `Atlas` device (or your name from step 2) from the Connect list.
   The credentials stay cached. The device then appears in the source
   list of Home Assistant.
5. In the admin webapp, open Plugins, then Home Assistant. Add the key
   `SPOTIFY_DEFAULT_SOURCE` and set it to the device name. A request that
   names a speaker still plays on that speaker.
6. To tune the behavior, add these keys to `/etc/atlas-edge/config.toml`:

   | Key | Default | Meaning |
   |---|---|---|
   | `music_fifo` | `/run/atlas-edge/music.fifo` | The FIFO path. An empty value turns music off. |
   | `music_duck_level` | `0.05` | Music volume while it is ducked, from 0.0 to 1.0. |
   | `music_duck_ramp_ms` | `150` | Time for the volume to change, in milliseconds. |

   Restart the edge service after you change a key.

The array plays audio at 16 kHz, so the music has no content above 8 kHz.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| The Pi's log shows an HTTP 403 on connect | The token is wrong, or an admin revoked it | Pair the device again (step 9) and rewrite `config.toml` (step 10) |
| The connection closes with code 4009 | Another Pi already holds this token's connection | Confirm only one Pi runs with this token, or pair a second device |
| The log shows "PortAudio library not found" | `libportaudio2` is not installed | Run step 2 again |
| `enforce` mode stops an enrolled member's turns | The threshold no longer fits this member's voice | Run `tune` again (step 9) with fresh labeled turns, or re-enroll the member (step 5) |
| `enforce` mode lets everyone through | No member is enrolled for the model `speaker_id.model` selects | Enroll at least one member (step 5), or select the model you enrolled members under |
| `atlas-librespot` restarts again and again, and its log shows "Avahi error: Setting up dns-sd failed" | The unit runs with `DynamicUser=yes`, and D-Bus cannot find that user | Use the unit from step 17, which runs as the `atlas-librespot` user. Create the user first (step 17, sub-step 1) |
| The Spotify device disappears from the Connect list, and the `atlas-librespot` log shows "Connection to server closed" | Spotify closed the session, and librespot 0.8.0 does not connect again. The process stays up, so systemd does not restart it | Install `atlas-librespot-watchdog` (step 17, sub-step 3). It restarts `atlas-librespot` when this line appears. To recover now, run `sudo systemctl restart atlas-librespot` |
