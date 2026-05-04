// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import './style.css'

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    enhanceAppWithTabs(app);
    // VitePress is an SPA: `<div hx-trigger="load">` placeholders only
    // fire on initial mount. After client-side navigation, new HTMX
    // placeholders need a manual `htmx.process(document.body)` to be
    // picked up. Hook the after-route-change event for that.
    if (typeof window !== 'undefined' && router) {
      router.onAfterRouteChanged = () => {
        // @ts-ignore - htmx loaded via head <script>; no types.
        if (window.htmx) window.htmx.process(document.body);
      };

      // Embedded BRM fragments contain root-absolute links like
      // `<a href="/pipeline?formula=…">` that point at the BRM
      // server's own paths. Without rewriting they resolve against
      // VitePress's origin and 404. Rewrite root-absolute hrefs and
      // hx-* URLs inside `.htmxo-embed` containers to go through the
      // `/live-brm` proxy.
      const rewritePrefix = (
        document.querySelector('meta[name="htmxo-embed-prefix"]')?.getAttribute('content')
        ?? '/live-brm'
      );
      const rewriteRootRefs = (root: HTMLElement) => {
        const fix = (el: Element, attr: string) => {
          const v = el.getAttribute(attr);
          if (v && v.startsWith('/') && !v.startsWith('//') && !v.startsWith(rewritePrefix)) {
            el.setAttribute(attr, rewritePrefix + v);
          }
        };
        root.querySelectorAll('[href]').forEach((el) => fix(el, 'href'));
        root.querySelectorAll('[hx-get]').forEach((el) => fix(el, 'hx-get'));
        root.querySelectorAll('[hx-post]').forEach((el) => fix(el, 'hx-post'));
      };
      document.body.addEventListener('htmx:afterSwap', (e: any) => {
        const tgt = e?.detail?.target as HTMLElement | undefined;
        if (!tgt) return;
        const embed = tgt.closest('.htmxo-embed');
        if (!embed) return;
        rewriteRootRefs(embed as HTMLElement);
      });
    }
  }
}
export default Theme
