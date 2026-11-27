# identity-service

One profile store and one operator UI over two existing identity engines: a
speaker-recognition HTTP service and a face-recognition HTTP service. The
service does not write model files itself — enrollment goes through each
engine's own API so the running engine sees the change. It also keeps a photo
library for the face training set, because the face engine stores embeddings
only and has no source frames to show.

This project is used by a voice gateway, but it is standalone: it has no code
dependency on the gateway, only an optional HTTP call to it for live camera
recording.

## Prerequisites

- Docker Engine with the Compose plugin.
- Linux host networking (`network_mode: host`). On Docker Desktop or a remote
  daemon, change the URLs in `.env` to reachable addresses instead.
- A speaker-id HTTP API with `/identify`, `/enroll` and `/health`
  (`SPEAKER_ID_URL`, default `http://localhost:8001`).
- A face HTTP API with `/recognize`, `/enroll`, `/health` and
  `DELETE /enroll/{name}` (`FACE_API_URL`, default `http://localhost:8090`).
- Optional: a voice gateway exposing `POST /api/identity/capture` for the
  "Record from camera" button (`GATEWAY_URL`, default `http://localhost:18792`).
  Without it, voice samples can still be uploaded as WAV files.

No model weights, photos, chat ids or profile data are included in this
repository.

## Run

```sh
cp .env.example .env
# Edit .env: set the four secrets and the URLs of your engines.
mkdir -p data speakers known_faces face_pending
docker compose up -d --build
```

The operator page is at <http://localhost:18794/> and asks for the Basic
credentials from `.env`.

Health check (unauthenticated, no secrets in the response):

```sh
curl -fsS http://localhost:18794/health
# {"status":"ok","admin":true,"token":true,"profiles":0}
```

`admin` and `token` report only whether the corresponding secret is set, never
its value. Empty secrets make the protected endpoints fail closed with 503.

Run the tests without Docker by installing `requirements.txt` and running
`pytest` from the repository root.

## Environment

All values come from one `.env` file; the compose file has no `env_file:`
entries and no absolute host paths.

| Variable | Default | Purpose |
|---|---|---|
| `IDENTITY_ADMIN_USER` | empty | Basic-auth user for the UI and all write endpoints. Empty disables them. |
| `IDENTITY_ADMIN_PASSWORD` | empty | Basic-auth password. Empty disables them. |
| `IDENTITY_TOKEN` | empty | Bearer token for read endpoints. Empty disables them. |
| `IDENTITY_CAPTURE_TOKEN` | empty | Bearer token sent to the gateway when recording from a camera. |
| `API_KEY` | empty | Optional `X-API-Key` header for speaker-id enrollment. |
| `SPEAKER_ID_URL` | `http://localhost:8001` | Speaker-recognition engine base URL. |
| `FACE_API_URL` | `http://localhost:8090` | Face-recognition engine base URL. |
| `GATEWAY_URL` | `http://localhost:18792` | Gateway base URL for `POST /api/identity/capture`. |
| `IDENTITY_ROOMS` | `room1,room2` | Deployment-specific room list for the picker and `/api/record` validation. Rename for your cameras. |
| `TZ` | `Europe/Moscow` | Container time zone. |

The compose file also sets `IDENTITY_DATA_FILE=/app/data/profiles.json`,
`VOICE_GALLERY_DIR=/app/speakers` and `FACE_PENDING_DIR=/app/face_pending`.
Override them only if you change the bind mounts.

## REST API

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness and whether secrets are configured. |
| `GET /` | Basic | Operator dashboard. |
| `GET /api/profiles` | Bearer or Basic | Profiles with gallery state and photo names. |
| `POST /api/profiles/{key}` | Basic | Create or edit a profile record. |
| `DELETE /api/profiles/{key}` | Basic | Delete the record only; engine samples are left alone. |
| `POST /api/profiles/{key}/voice/enroll` | Basic | Upload voice samples; reports measured sample quality. |
| `POST /api/profiles/{key}/face/enroll` | Basic | Upload photos; keeps originals for enrollment and crops as thumbnails. |
| `DELETE /api/profiles/{key}/face/photos/{name}` | Basic | Remove one photo and rebuild the embedding from the rest. |
| `POST /api/profiles/{key}/face/curate` | Basic | Keep the best frames up to the configured cap. |
| `GET /api/photos/{key}/{name}` | Bearer or Basic | Serve one training thumbnail. |
| `POST /api/identify` | Bearer or Basic | Speaker identification joined with the profile. |
| `POST /api/recognize` | Bearer or Basic | Face recognition joined with the profiles. |
| `POST /api/record` | Basic | Record a WAV from a camera and replace the voice sample. |

Enrollment is destructive: voice `enroll` replaces the stored centroid; there
is no append mode. The UI says so before each upload.

## License

Dual-licensed under [GNU AGPL v3 or later](LICENSE) or a paid
[Commercial License](COMMERCIAL-LICENSE.md). Pick the one that fits your use.
