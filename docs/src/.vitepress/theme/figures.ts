// The build copies AlgebraOfVega's versioned vendor bundle. Its runtime owns
// Vega views, neutral inks, light/dark updates, resizing and disposal.
declare global {
  interface Window {
    AoV?: { embed: (id: string, spec: object, options: object) => unknown }
  }
}

export function setupFigures() {
  const rendered = new WeakMap<HTMLElement, string>()
  let nextId = 0
  let queued = false

  function showError(element: HTMLElement, error: unknown) {
    const message = document.createElement('p')
    message.setAttribute('role', 'alert')
    message.textContent = `Figure could not be rendered: ${String(error)}`
    element.replaceChildren(message)
    console.error('BRM docs figure', error)
  }

  function scan() {
    queued = false
    for (const marker of document.querySelectorAll<HTMLElement>('div.language-brm-figure')) {
      try {
        const payload = JSON.parse(marker.querySelector('pre code')?.textContent || '')
        if (typeof payload.spec !== 'string' || typeof payload.caption !== 'string' ||
            typeof payload.alt !== 'string') throw new Error('Invalid figure payload')
        const figure = document.createElement('figure')
        figure.className = 'brm-figure'
        const plot = document.createElement('div')
        plot.className = 'brm-figure-plot'
        plot.dataset.brmVega = payload.spec
        plot.setAttribute('aria-label', payload.alt)
        const status = document.createElement('p')
        status.setAttribute('role', 'status')
        status.textContent = 'Loading figure…'
        plot.appendChild(status)
        const caption = document.createElement('figcaption')
        caption.textContent = payload.caption
        figure.append(plot, caption)
        marker.replaceWith(figure)
      } catch (error) {
        marker.classList.remove('language-brm-figure')
        showError(marker, error)
      }
    }
    if (!window.AoV) return
    for (const element of document.querySelectorAll<HTMLElement>('[data-brm-vega]')) {
      const json = element.dataset.brmVega!
      if (rendered.get(element) === json) continue
      rendered.set(element, json)
      element.id ||= `brm-figure-${nextId++}`
      try {
        const spec = JSON.parse(json)
        void Promise.resolve(window.AoV.embed(element.id, spec, {
          actions: false, renderer: 'svg',
        })).catch(error => showError(element, error))
      } catch (error) {
        showError(element, error)
      }
    }
  }

  function schedule() {
    if (queued) return
    queued = true
    requestAnimationFrame(scan)
  }

  new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true })
  document.addEventListener('load', schedule, true)
  document.addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement && /vega|aov-runtime/.test(event.target.src)) {
      document.querySelectorAll<HTMLElement>('[data-brm-vega]').forEach(element => {
        if (!rendered.has(element)) showError(element, 'Figure library could not be loaded. Reload to retry.')
      })
    }
  }, true)
  scan()
}
