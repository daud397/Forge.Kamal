# Listing Forge

A working prototype of the pipeline in your diagrams: CAD files and product
photos go in, marketplace-ready images come out, with the product's colour and
geometry protected along the way.

```
upload ──► render CAD ──► Astra writes the brief ──► Flare drafts 3 scenes
                                                            │
                                    you pick one ───────────┘
                                          │
                     Sunburst masked edit ─┴─► composite ──► measured QA
                                                                  │
                                              Lanczos upscale ────┴──► exports
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
./run.sh                      # http://127.0.0.1:8000
```

Try it without spending anything first:

```bash
MOCK=1 ./run.sh
```

Mock mode runs every stage with synthesised images and makes no API calls. The
whole UI is clickable, so you can see the shape of the thing before you decide
whether it's worth building properly. Note that mock mode deliberately shifts
the product between stages, which makes the QA step fail — that's the QA doing
its job, not a bug.

STEP and IGES need one extra package: `pip install cascadio`. Everything else
(STL, OBJ, PLY, GLB, 3MF) works out of the box.

## What each stage actually does

**Render.** The renderer is pure numpy — no GPU, no OSMesa, no Blender install.
Three camera angles per model. The reason to render rather than photograph is
the alpha channel: the silhouette comes straight out of the depth buffer, so the
mask handed to the editing model is exact. Segmenting a product out of a
photograph is guesswork by comparison. Roughly 0.2s per 1024px frame on a
low-poly test model; meshes above 80k faces get decimated first.

**Brief.** `gpt-6-astra` reads the renders and returns a structured JSON brief —
product, materials, sampled colours as hex, what must survive editing, and three
scene prompts. The scene prompts describe the *background only*. The product is
never described, because it's composited in unchanged.

**Drafts.** `gpt-image-2.5-flare`, one call per scene, 1024×1024 at medium
quality. Cheap enough to throw away, which is the point of a draft.

**Final.** `gpt-image-2.5-sunburst` at 1536×1536, high quality, with the mask.
Then the product pixels are composited back from the source. This is deliberate:
OpenAI's own guidance says that if a region must stay pixel-identical you should
composite rather than rely on prompting, because repeated edits drift. So the
model supplies the background and your source file supplies the product.

**QA.** Two halves, deliberately separated:

- *Measured* — CIEDE2000 colour difference and SSIM over the product region,
  computed in numpy. Verified against the standard Sharma test vectors. The
  measurement grades the **raw model output**, not the composited result;
  grading the composite would grade the fix and every run would pass.
- *Judged* — Astra looks at the composited image and is asked only about things
  numbers can't catch: does the lighting on the product match the background, do
  the contact shadows sit right, is there stray text or a duplicated object.

A vision model asked "is this colour correct?" will agree with you far too
readily. CIEDE2000 will not.

**Export.** Lanczos only. No second generative pass, so the upscale can't invent
detail that wasn't there. Generation deliberately stays under OpenAI's
3,686,400-pixel experimental threshold and the resize carries it the rest of the
way — cheaper and more predictable than generating at 4K.

## Driving it by chat

The right-hand panel takes plain instructions instead of clicks. Astra is given
the current job state and a set of tools that map onto the same pipeline stages
the buttons call, so a run driven by conversation ends up in exactly the same
place as one driven by clicking.

What it understands:

| You say | What happens |
|---|---|
| "put it on a marble counter with morning window light" | writes new scene prompts, regenerates drafts |
| "give me three options, one warm, one clinical, one outdoors" | three scenes at once |
| "go with the second one but shorten the shadow" | picks that draft, runs the masked final edit |
| "make the key light softer" | re-runs the final on the same draft |
| "export for shopify and etsy" | renders those presets |
| "why did the quality check fail?" | explains the measured numbers |

It will refuse things the pipeline genuinely can't do rather than attempting
them - adding text or logos, changing the product itself.

In mock mode there's no model behind the chat, so a keyword parser stands in. It
handles the instructions above and tells you plainly when it doesn't understand,
rather than guessing. Everything conversational only works once you're live.

One rule the agent is held to: scene prompts describe the **background only**,
never the product. Describing the product invites the model to redraw it, and
the composite would throw that work away regardless.

## One thing to decide before you use this on real listings

`amazon_main` is flagged `generative: false` and the UI marks it in amber.
Amazon requires the main image to be an actual photograph of the product. A
rendered or AI-composed hero image in that slot risks suppression. Lifestyle and
secondary slots are far more permissive, and your own storefront has no such
rule.

The pipeline will happily produce a 2000×2000 white-background image for that
slot. The flag exists so nobody uploads it without having made that call
deliberately.

## Layout

Everything sits at the root. There are no subfolders, deliberately: GitHub's
web uploader silently flattens directory structure, and a flat project can't be
broken by that.

```
config.py     models, tolerances, size rules — everything tunable
cad.py        mesh loading and the software rasterizer
imaging.py    CIEDE2000, SSIM, masks, compositing, premultiplied resizing
models.py     all OpenAI traffic, plus mock mode
agent.py      the chat interface and its tools
auth.py       password gate
budget.py     spend tracking and the daily cap
pipeline.py   the stages
store.py      SQLite job state
main.py       HTTP routes
index.html / styles.css / app.js    frontend, no build step
```

The three frontend files are served by name rather than by mounting the
directory. A static mount at the root would sit on top of the source tree and
serve `auth.py` to anyone who guessed the filename.

Run locally with `uvicorn main:app --port 8000` — note `main:app`, not
`app.main:app`.

## Things to know before scaling this up

**Mask polarity.** OpenAI's `images.edit` treats transparent pixels as the
editable region. Some third-party gateways invert it. If your first masked edit
comes back with the background untouched and the product mangled, flip
`MASK_TRANSPARENT_IS_EDITABLE=false`.

**Rate limits and cost.** Jobs run on a two-worker thread pool, which is fine
for one person and wrong for a team. Swap in Redis and a real queue before more
than a couple of people use it at once. Nothing tracks spend yet — worth adding
before a batch run.

**Renderer coverage.** The rasterizer handles solid geometry with per-face flat
shading. It has no materials, no textures, no transparency, and no support for
CAD assemblies with distinct part colours. For a first pass into a generative
model that's sufficient, because the render is scaffolding rather than the
deliverable. If you find the model misreading a product's finish, that's the
first place to invest.

**Pricing wasn't hardcoded anywhere**, because per-image pricing for the 2.5
family was still moving when this was written. Check the dashboard before a
large batch.


## Putting it on a VPS

Hostinger supports Python only on their VPS plans, not shared or web hosting,
because it needs root to install Python and its dependencies. Anything below VPS
cannot run this.

`deploy/setup.sh` does the whole server build on a fresh Ubuntu 22.04 or 24.04
box: system packages, a service account that cannot log in, a virtualenv, a
systemd unit that survives reboots, Nginx in front, a firewall, and a Let's
Encrypt certificate.

```bash
# on the VPS, as root
unzip listing-forge.zip -d /opt
bash /opt/listing-forge/deploy/setup.sh yourdomain.com
nano /opt/listing-forge/.env        # paste OPENAI_API_KEY
systemctl restart forge
```

It generates and prints a portal password. Save it when it appears; it is also
in `.env`, which is readable only by root.

Point your domain's A record at the VPS IP before running the script, or the
certificate step fails. Everything else still works and you can rerun certbot
later.

### Two things the deployment adds

**A password gate.** One shared password, an HMAC-signed cookie, no user
accounts. Unauthenticated API requests get 401 and page requests get bounced to
a login form. Repeated wrong guesses from one address get damped after eight
attempts.

With `PORTAL_PASSWORD` unset there is no gate at all. That's correct on
localhost and dangerous anywhere else, which is why the setup script always
generates one.

**A daily spend cap.** Every model call is checked against `DAILY_CAP` before it
is made, and a batch is refused whole rather than partway through. The header
shows the running total, amber at 80% and red at the ceiling.

The cap is enforced against *estimated* costs from `.env`, not billed amounts.
Run a handful of real jobs, compare against your OpenAI dashboard, and correct
`COST_PER_IMAGE` — otherwise the cap protects you by the wrong margin.

### Running it

```bash
systemctl status forge          # is it up
journalctl -u forge -f          # live logs
systemctl restart forge         # after editing .env
```

`.env` changes need a restart; nothing reads them at runtime.

### What this deployment is not

Single box, SQLite, a thread pool. Fine for a few people sharing a tool. It has
no horizontal scaling, no backups beyond whatever your host snapshots, and no
per-user audit trail — if you need to know which colleague ran which job, the
shared password has to be replaced rather than extended.


## Deploying on Railway instead

Less to set up and less to get wrong than a VPS: no SSH, no Nginx, no
certificates, no firewall. `Dockerfile` and `railway.json` are included.

```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

Then in the Railway dashboard:

1. **Variables** - add `OPENAI_API_KEY`, `PORTAL_PASSWORD`, `SESSION_SECRET`
   (any long random string), `HTTPS=1`, and `DAILY_CAP`.
2. **Volumes** - attach one mounted at `/app/data`. This is not optional. The
   container filesystem is wiped on every redeploy, so without a volume every
   job, render and export disappears the moment you push a change.
3. **Settings → Networking** - generate a domain. HTTPS is automatic.

Redeploy after adding the variables; nothing reads them at runtime.

### Railway against a VPS

Railway bills by usage rather than a flat monthly fee, so an idle portal costs
little and a heavy render day costs more. A VPS is a fixed price whatever you do
with it, and gives you a box you can SSH into when something misbehaves.

The CAD rasterizer is CPU-bound and single-threaded. On either platform, a dense
production mesh takes meaningfully longer than the low-poly test model - worth
measuring on your own files before deciding how much machine to pay for.
