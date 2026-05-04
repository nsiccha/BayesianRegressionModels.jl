---
aside: false
outline: false
---

<style>
/* Break the gallery embed out of VitePress's narrow content column. */
.VPDoc:has(.htmxo-embed) > .container > .content { max-width: none !important; }
.VPDoc:has(.htmxo-embed) .content-container { max-width: none !important; }

/* Override BRM's inline gallery grid with a docs-friendly auto-fit
 * layout: as many cards per row as fit at >=420px each. Drops to 1
 * col on phones. */
.htmxo-embed .brm-gallery-grid {
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)) !important;
    gap: 0.75rem !important;
}
.htmxo-embed article {
    overflow: visible !important;
    min-width: 0;
}
.htmxo-embed article > div { overflow-x: auto; }
</style>

# Gallery

BRM ships a gallery of presets and worked examples. Each card embeds
the input formula, the SLIC submodel body emitted by sbimpl, the
transpiled Stan source, and the auto-generated posterior-predictive
check.

::: tip Live and recorded
- **Local app:** [`http://localhost:8121`](http://localhost:8121) once
  started via `~/github/nsiccha/Claude/start-web.sh BRMMacro` (or
  `julia --project=web/app -i web/app/main.jl 8121`).
- **Refresh the deploy recordings:** visit
  `http://localhost:8121/record_gallery` in a browser. The route runs
  the full pipeline (Stan compile + Pathfinder fit) for each preset
  and example, dumps every gallery URL into `docs/src/public/live-brm/`,
  then `git add docs/src/public/live-brm && git commit && git push`
  — CI picks up the recordings as static assets.
:::

## Live preview

The BRM gallery rendered inline below — fetched via HTMX
(`HX-Request: true` → BRM's `__page__` returns a body fragment that
drops into this page). VitePress proxies `/live-brm/*` to the running
BRM server (`BRM_DEV_TARGET=http://localhost:8121` by default) in dev,
and to recordings in production.

<div class="htmxo-embed" data-hx-base="live-brm/pipeline/gallery" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading BRM gallery…</em>
</div>

<script>
// Make the embed URL base-aware. In dev `import.meta.env.BASE_URL` is
// `/`, so `data-hx-base="live-brm/pipeline/gallery"` becomes
// `/live-brm/pipeline/gallery` (matches Vite's proxy).
// In prod the base is `/BayesianRegressionModels.jl/dev/`, so it
// becomes `/BayesianRegressionModels.jl/dev/live-brm/pipeline/gallery`
// (matches the committed recordings). Same markdown works in both
// deploys.
//
// Repeated on every VitePress route change because the SPA may render
// the page after this script's initial run; idempotent on re-entry.
(function () {
  function rewrite() {
    const base = (typeof __DEPLOY_ABSPATH__ !== "undefined" && __DEPLOY_ABSPATH__) || "/";
    document.querySelectorAll("[data-hx-base]").forEach((el) => {
      if (el.hasAttribute("hx-get")) return;
      el.setAttribute("hx-get", base.replace(/\/$/, "") + "/" + el.getAttribute("data-hx-base"));
      if (window.htmx) window.htmx.process(el);
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", rewrite);
  } else {
    rewrite();
  }
})();
</script>
