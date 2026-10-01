# The ATLAS Mac app

The ATLAS Mac app is a menu bar app. It pairs with your ATLAS server, keeps a live connection, and shows the state of that connection in the menu bar.

The app shows a panel with the live transcript and a ringing timer. It does not move windows or run actions. Pairing, the connection, the panel, and the permissions it will need later are the only parts that work today.

This guide is a procedure. Every host in it is invented. Replace `atlas.example.com` with the host of your own server. Do not put a real hostname, token, or signing identity into this file or into a copy of it.

## Prerequisites

1. Use an Apple silicon Mac that runs macOS 15 or later.
2. Install full Xcode from the App Store. The Command Line Tools alone are not enough.
3. Open Xcode once and accept the license.
4. Make sure the active developer folder is Xcode, not the Command Line Tools:

   ```bash
   xcode-select -p
   ```

   The path must end with `.app/Contents/Developer`, for example `/Applications/Xcode.app/Contents/Developer` or `/Applications/Xcode-16.4.app/Contents/Developer`. The name of the app can differ. A path that ends with `CommandLineTools` is not enough. If you see that path, open Xcode, then open Settings, Locations, and choose your Xcode in the Command Line Tools list.
5. Use an ATLAS server that you reach over https, for example `https://atlas.example.com`. The app connects over wss only. It has no setting that allows a plain connection.
6. Have an admin account on that server. You need it to add a Mac.

## Install

1. Clone this repository on the Mac:

   ```bash
   git clone <repository url>
   cd atlas
   ```

2. Run the install script:

   ```bash
   desktop/scripts/install.sh
   ```

3. Enter your login password when macOS asks. The first run asks for it to make a signing certificate. See [Signing identity](#signing-identity).
4. Wait for the build. The script signs the app, copies it to `/Applications/ATLAS.app`, and opens it.

The script checks this Mac first. It stops with a clear message if the macOS version, the chip, or Xcode does not fit.

A second run is a rebuild. It keeps the same signing identity. Run it again after you update the repository.

The script writes to `/Applications`, to its own `desktop/build` and `desktop/.build` folders, and to `desktop/.signing-identity`. If it makes the local certificate, it also adds that certificate, its private key, and a code-signing trust setting to your login keychain. It never asks for administrator rights. If your account cannot write to `/Applications`, run the script from an administrator account.

To read all options, run `desktop/scripts/install.sh --help`.

## Signing identity

macOS keeps two things for an app: the Accessibility grant and the token in the Keychain. It keeps them only while the app signing identity stays the same. If the identity changes, macOS resets both. You then grant Accessibility again and pair again.

The install script picks the identity in this order:

1. The `ATLAS_SIGN_IDENTITY` environment variable. It holds a SHA-1 hash or an exact identity name.
2. The identity that the script saved in `desktop/.signing-identity`. Git does not track this file.
3. The first Developer ID Application identity in your keychain.
4. The first Apple Development identity in your keychain.
5. A local certificate named "ATLAS Local Signing". The script makes it once.

The script signs by SHA-1 hash. It never signs ad hoc.

### Recommended: an Apple Development certificate

A free Apple Development certificate gives the best result. It carries a team id. macOS then keeps the Keychain item across rebuilds.

1. Open Xcode.
2. Open Settings, then Accounts.
3. Add your Apple ID. A free Personal Team is enough.
4. Select the team and click Manage Certificates.
5. Click the plus button and choose Apple Development.
6. Run `desktop/scripts/install.sh`.

The script finds the certificate and uses it. You do not need a paid developer account.

If you already ran the script with the local certificate, the saved identity in `desktop/.signing-identity` wins. To switch, find the hash of the new certificate:

```bash
security find-identity -v -p codesigning
```

Then run the script with that hash:

```bash
ATLAS_SIGN_IDENTITY=<identity hash> desktop/scripts/install.sh
```

The script saves the new identity. The change resets the grants once. Grant Accessibility again and pair again.

### Fallback: the local certificate

If your keychain has no other identity, the script makes the "ATLAS Local Signing" certificate. This certificate has no team id. It works, with one cost: macOS asks for Keychain access again after each rebuild. The Accessibility grant is not affected.

The first run asks for your login password twice:

1. A macOS dialog asks you to trust the new certificate for code signing.
2. The terminal then asks for the login password. This step lets `codesign` use the new key without a prompt on each build.

The script never holds your password. The `security` tool asks for it. The script puts no password on a command line. It deletes the private key file as soon as the key is in your keychain.

After each later rebuild, macOS asks once in a Keychain dialog. Choose Always Allow. The app then reads its token with no further prompt until the next rebuild.

If you made the certificate with an older version of the script, delete it and make it again. The older certificate does not have the key access fix, and `codesign` asks for access on every build.

1. Open Keychain Access.
2. Search for "ATLAS Local Signing".
3. Delete the certificate and the private key that belongs to it. Open the My Certificates tab and expand the certificate to find the key. Keychain Access names that key "Imported Private Key".
4. Run `desktop/scripts/install.sh`. The script makes a new certificate.

A new certificate is a new identity. The grants reset once.

### What is proven

The signing check ran on macOS 26 with the local certificate. These results hold:

- The designated requirement stayed the same across three rebuilds.
- The Accessibility grant survived all three rebuilds.
- `install.sh --verify` exited 0 after each rebuild.
- The Keychain asked once after each rebuild.

These parts are not yet proven:

- The Developer ID Application row and the Apple Development row. The test Mac had no Apple identity. The Keychain item is expected to survive rebuilds without a prompt, because the identity carries a team id. A person must confirm this on a real Mac.

The app is not notarized in this phase. A build from your own Mac has no quarantine flag, so macOS runs it without notarization.

### Check the identity

Run this command to compare the signing identity of a new build with the installed app:

```bash
desktop/scripts/install.sh --verify
```

The script builds into `desktop/build` and installs nothing. It exits 0 when the identity matches the installed app. It exits 1 when the identity differs or when no app is installed.

Never commit an identity, a hash, or a team id. The file `desktop/.signing-identity` stays on your Mac.

## Permissions

The setup window opens at first launch. It has five steps. Pair, Accessibility, and Local Network are required. Continue stays off until all three are done.

| Step | Required | What it is for |
| --- | --- | --- |
| Pair | Yes | Connects this Mac to your ATLAS server. |
| Accessibility | Yes | ATLAS needs this to move and size windows later. It does nothing with it yet. |
| Local Network | Yes | ATLAS needs this to reach a server on your home network. |
| Location and Set home | No | A rough location tells Home from Away in the menu bar. The location stays on this Mac and the app never sends it to the server. |
| Launch at login | No | Opens ATLAS when you log in, so it stays connected. It is on by default. |

### Accessibility

1. Click Open Settings on the Accessibility row.
2. Turn on ATLAS in System Settings, Privacy and Security, Accessibility.

The row changes to Granted a moment later. You do not need to restart the app.

### Local Network

macOS has no way to ask whether this permission is on. The app infers the state from the connection:

- Checking… means no connection attempt has finished yet.
- Granted means the server answered. A server that refuses the connection also counts, because it proves the network path works.
- Not granted means the first connection to a host on your home network failed.

A past successful connection stays Granted, so a server that is down never looks like a missing permission.

If the row stays Checking…:

1. Make sure the Mac is paired and the server is up. Open `https://atlas.example.com` in a browser.
2. Wait for the next connection attempt. The row changes on its own.
3. If you use a public host name, the app has no failure it can blame on Local Network. Check the host name and the network instead.

If the row shows Not granted:

1. Click Open Settings on the Local Network row.
2. Turn on ATLAS in System Settings, Privacy and Security, Local Network.
3. Wait for the next connection attempt. The row changes to Granted.

### Location

Location is optional. If you allow it, open the ATLAS menu and choose Set Home Here when you are at home. The menu bar then shows Away, not Offline, when the Mac is offline and more than 1 km from home. If you deny it, ATLAS works as before and the menu bar shows Offline.

### Launch at login

This step is on by default. macOS may ask you to approve the login item. If the row shows Needs approval, click Open Login Items and turn on ATLAS in System Settings, General, Login Items.

### Notifications

The app sends one notification. It reads "ATLAS was unpaired" and appears when the server revokes this Mac. macOS may ask you to allow it.

## Pairing

1. Open the ATLAS webapp and sign in as an admin.
2. Open Macs and click Add Mac. Enter a name for the Mac. Names are unique, and the check ignores case.
3. Copy the pair link or the token. The webapp shows both once. It does not show them again, even after a reload.
4. On the Mac, click Open in ATLAS, or paste the pair link into the Pair step of the setup window.
5. To enter the values by hand, open "Enter a server and token instead" and fill in Server and Token. Use a host such as `atlas.example.com`.
6. Read the server host in the confirm dialog. Pair only with a server that you run.
7. Click Pair. The Pair step changes to Paired with the host.

A pair link looks like this: `atlas://pair?server=atlas.example.com&token=<token>`.

If the Mac is already paired, a second link shows "Replace the current pairing?". Choose Replace or Keep Current.

After the Mac connects, the Macs page shows Online. Click Test to check that the Mac answers. The button stays off while the Mac is offline.

On the Macs page you can also:

- Pick a Room for the Mac.
- Turn on Default Mac. One Mac at a time is the default.
- Click Revoke to remove the Mac. The app shows Revoked and stops its retries.

A Mac that never connected shows "Never connected".

### The menu bar

The first line of the menu shows the state.

| Menu line | Meaning |
| --- | --- |
| Not paired | No pairing. Open Setup and pair. |
| Connecting to the host… | The app is dialing the server. |
| Connected to the host, since a time | The link is up. The globe icon is filled. |
| Offline, with the host and the last connection time | The server is not reachable. The app keeps trying. |
| Away, with the host and the last connection time | The Mac is offline and more than 1 km from home. |
| Revoked, pair again | The server refused the token. Pair again. |

A version mismatch between the app and the server also reads as Offline. Run `desktop/scripts/install.sh` again after you update the repository.

The menu has these items: Stop Ringing while a timer rings, Finish Setup… while a required step is open, Pair… when the Mac is not paired or revoked, Set Home Here when Location is allowed, Setup… when the required steps are done, and Quit ATLAS. Stop Ringing is first, below the state line. See [Panel](#panel).

## Panel

The panel is a small window that shows what ATLAS hears and says. It also shows a ringing timer. The panel needs a paired Mac that is online.

### What opens the panel

Two things open the panel:

- A wake that the server confirms. The server confirms a wake when the transcript starts with the wake phrase. A hit from the wake-word detector alone opens nothing.
- A timer or an alarm that rings. No wake word is needed.

A wake can come from any microphone in the house. The panel opens on every online Mac, not only on the Mac that is near the microphone.

A source whose transcript is not checked for the wake phrase opens no panel. The browser listen pages open no panel.

### Where it shows

The panel shows in the top-right corner of the display that has the pointer. It shows over full-screen apps and on every Space. It never takes the keyboard focus. You can keep typing in the app that you use.

### What it shows

The panel shows these items:

- The live transcript of what you say. The final transcript does not include the wake phrase.
- The state line: Listening, Thinking, Speaking, or Done.
- The spoken reply, as plain text.
- A short fixed line when ATLAS did not catch the request.

The panel shows all text as plain text. A link in a transcript is not a link.

### When it hides

The panel hides about 4 seconds after the reply audio ends. If ATLAS asks a question, the panel stays open for the follow-up window. Put the pointer on the panel to keep it open. Click Close to hide it at once. The panel also hides when the connection drops.

### Who sees the transcript

Every online Mac shows the transcript of every confirmed wake. This includes a Mac that other people use. This version has no switch for each Mac. Pair only Macs that you want to show these transcripts.

The panel opens before ATLAS checks who is speaking. If a television or a voice that ATLAS does not know says the wake phrase, the panel shows what it said. ATLAS then refuses the request, and the panel closes with "Did not catch that". The panel shows this text even though ATLAS refused the request. A sentence that does not start with the wake phrase shows nothing.

### Timers and alarms

When a timer or an alarm rings, the panel opens with a bell, the word Timer or Alarm, and the label. If the label is empty, the panel shows "Time is up" or "Alarm is ringing". A Mac that connects while a timer rings shows the ring at once.

To stop the ring, use one of these ways:

1. Click Stop in the panel. The first click works while another app is active. The button shows "Stopping…", then the panel shows "Stopped".
2. Open the ATLAS menu and choose Stop Ringing. This item shows only while a timer rings. It works after you click Close, because Close hides the panel and does not stop the sound.
3. Say "stop".

All three ways stop the ring on every Mac. If the server does not answer in 3 seconds, the Stop button turns on again. Click it again.

A ring takes over the panel while a turn is open. The turn comes back when the ring stops.

### Accessibility

VoiceOver reads an announcement when the panel opens, when a card shows, and when a timer rings. The ring announcement is "Timer ringing" or "Alarm ringing", then the label. Reduce Motion turns off the fades, the pulse on the state symbol, and the ripple on the ring graphic.

## Troubleshooting

**The menu says Offline but the server is up.** The Local Network permission may be off. Open the setup window, find the Local Network row, and click Open Settings. Turn on ATLAS. The app connects again by itself. See [Local Network](#local-network).

**The menu says Offline behind a reverse proxy.** A proxy, firewall, or ingress can answer the connection with an HTTP 403 of its own. The app does not treat that as a refused token, so it keeps the pairing and tries again. The app reads a token refusal only from a 403 that carries the `X-Atlas-Refusal: token` header, which only the ATLAS server sends. Make sure the proxy forwards the `Authorization` header and the WebSocket upgrade, and passes the server's response headers through unchanged.

**The Local Network row stays on Checking….** The app shows a result only after a connection attempt ends. Open the server in a browser to make sure it is up. Wait for the next attempt. See [Local Network](#local-network).

**Accessibility resets after a rebuild.** The signing identity changed. Run `desktop/scripts/install.sh --verify`. It exits 1 when the new build has a different identity from the installed app. Then pin the identity with `ATLAS_SIGN_IDENTITY`. See [Signing identity](#signing-identity). Grant Accessibility again after you pin it.

**The Keychain asks for access after each rebuild.** The app uses the local certificate, which has no team id. This is the known cost of the fallback. Choose Always Allow. To remove the prompt, use an Apple Development certificate. See [Recommended: an Apple Development certificate](#recommended-an-apple-development-certificate).

**Every build asks "codesign wants to sign using key".** The certificate came from an older version of the script. Delete "ATLAS Local Signing" in Keychain Access and run `desktop/scripts/install.sh` again. See [Fallback: the local certificate](#fallback-the-local-certificate).

**The certificate is lost.** You deleted "ATLAS Local Signing", or you restored the Mac from a backup. The next install makes a new certificate. This is a new identity, so macOS resets the grants once. Grant Accessibility again and pair again.

**The Accessibility grant is stuck.** Reset it, then grant it again:

```bash
tccutil reset Accessibility org.atlas-assistant.desktop
```

Open the setup window and click Open Settings on the Accessibility row.

**"The server did not accept this token."** The server revoked the token, or the token has a typing error. Add a new Mac on the Macs page and pair again.

**"This link is not secure."** The pair link starts with a plain scheme. The app pairs over wss only. Open the Macs page over https and copy a new link.

**"This app and the server use different versions."** Update the repository and run `desktop/scripts/install.sh` again.

**"Could not save the token to Keychain."** Unlock the login keychain and pair again.

**The server uses a certificate from an internal certificate authority.** Trust that authority on the Mac. Add its certificate to Keychain Access and set it to Always Trust. The app never skips certificate checks and has no setting that does.

### Collect facts for a bug report

Run the app in diagnose mode. The command waits until the app exits:

```bash
open -n -W /Applications/ATLAS.app --args --diagnose
```

Then read the last line of the launch log:

```bash
tail -n 1 ~/Library/Logs/ATLAS/launch.jsonl
```

The line holds the Accessibility state (`ax_trusted`), the Keychain read status, the login item status, and the signing requirement. It never holds the token. It does hold the host that the Mac is paired with. Remove that value before you share the line.

To test one server from this Mac, add `--host atlas.example.com`. The command accepts a bare host or `host:port`. Add `--write-probe` to store a test item in the Keychain, so a later rebuild can show whether access survived.

## Check that permissions survive a rebuild

Do this check once on a new Mac. It proves that rebuilds keep the Accessibility grant and the Keychain token.

1. Pair the Mac and grant Accessibility.
2. Run `desktop/scripts/install.sh`. The script opens the app when it ends.
3. Read the last line of the launch log:

   ```bash
   tail -n 1 ~/Library/Logs/ATLAS/launch.jsonl
   ```

4. Check these two values: `ax_trusted` must be `true`. `keychain_pairing_status` must be `0`.
5. Run `desktop/scripts/install.sh --verify`. It must exit 0.
6. Repeat steps 2 to 5 two more times. That makes three rebuilds in a row.
7. Open the Macs page. After each rebuild the Mac must show Online again within seconds.

With an Apple Development or Developer ID identity, no Keychain or Accessibility dialog appears during the three rebuilds.

With the local certificate, the Keychain asks once after each rebuild. Choose Always Allow. Read the log again after you answer. The status is then `0`. A status other than `0` before you answer is expected. The field `keychain_pairing_outcome` then shows `needs_interaction`.

## Uninstall

1. Quit ATLAS from the menu bar.
2. Open System Settings, General, Login Items. Remove ATLAS from the list.
3. Delete the app:

   ```bash
   rm -rf /Applications/ATLAS.app
   ```

4. Delete the Keychain items. The first holds the pairing token. The second holds a test item:

   ```bash
   security delete-generic-password -s org.atlas-assistant.desktop.pairing
   security delete-generic-password -s org.atlas-assistant.desktop.diagnostics
   ```

5. Reset the Accessibility grant:

   ```bash
   tccutil reset Accessibility org.atlas-assistant.desktop
   ```

6. Delete the launch log:

   ```bash
   rm -rf ~/Library/Logs/ATLAS
   ```

7. Open the Macs page in the webapp and revoke the Mac.

To remove the signing certificate too, delete "ATLAS Local Signing" and its private key in Keychain Access. The next install makes a new certificate and a new identity.
