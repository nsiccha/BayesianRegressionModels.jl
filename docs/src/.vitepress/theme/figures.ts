type VegaResult = { view: { finalize(): void } }
type VegaEmbed = (element: HTMLElement, spec: object, options: object) => Promise<VegaResult>

declare global {
  interface Window { vegaEmbed?: VegaEmbed }
}

// Documenter emits inert JSON attributes, so Vue hydration and client-side
// navigation both use the same renderer. Each chart owns its Vega view.
export function setupFigures() {
  const views = new Map<HTMLElement, { dark: boolean; result?: VegaResult }>()
  let queued = false

  function scan() {
    queued = false
    for (const [element, state] of views) {
      if (!element.isConnected) {
        state.result?.view.finalize()
        views.delete(element)
      }
    }
    const embed = window.vegaEmbed
    if (!embed) return
    const dark = document.documentElement.classList.contains('dark')
    for (const element of document.querySelectorAll<HTMLElement>('[data-brm-vega]')) {
      const previous = views.get(element)
      if (previous?.dark === dark) continue
      previous?.result?.view.finalize()
      const state: { dark: boolean; result?: VegaResult } = { dark }
      views.set(element, state)
      const ink = dark ? '#e6e6e6' : '#0b0b0b'
      const grid = dark ? '#3c3c43' : '#e2e2e3'
      try {
        const spec = JSON.parse(element.dataset.brmVega!)
        const authored = spec.config || {}
        const chrome = { labelColor: ink, titleColor: ink }
        spec.background = 'transparent'
        spec.config = {
          ...authored,
          font: getComputedStyle(element).fontFamily,
          axis: { ...chrome, gridColor: grid, tickColor: grid, domainColor: grid, ...authored.axis },
          legend: { ...chrome, ...authored.legend },
          header: { ...chrome, ...authored.header },
          title: { color: ink, ...authored.title },
          view: { stroke: grid, ...authored.view },
          range: { category: ['#2a78d6', '#eb6834'], ...authored.range },
        }
        void embed(element, spec, { actions: false, renderer: 'svg' }).then(result => {
          if (!element.isConnected || views.get(element) !== state) result.view.finalize()
          else state.result = result
        }).catch(error => showError(element, state, error))
      } catch (error) {
        showError(element, state, error)
      }
    }
  }

  function showError(element: HTMLElement, state: { dark: boolean }, error: unknown) {
    if (!element.isConnected || views.get(element) !== state) return
    const message = document.createElement('p')
    message.setAttribute('role', 'alert')
    message.textContent = `Figure could not be rendered: ${String(error)}`
    element.replaceChildren(message)
    console.error('BRM docs figure', error)
  }

  function schedule() {
    if (queued) return
    queued = true
    requestAnimationFrame(scan)
  }

  new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true })
  new MutationObserver(schedule).observe(document.documentElement, {
    attributes: true, attributeFilter: ['class'],
  })
  // The existing theme loads Vega asynchronously, in dependency order.
  document.addEventListener('load', schedule, true)
  document.addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement && /vega/.test(event.target.src)) {
      document.querySelectorAll<HTMLElement>('[data-brm-vega]').forEach(element => {
        if (!views.has(element)) element.textContent = 'Figure library could not be loaded. Reload to retry.'
      })
    }
  }, true)
  scan()
}
