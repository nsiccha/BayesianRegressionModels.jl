// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import { onMounted } from 'vue'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import { setupBackendComparisons } from './backend-comparison'
import { setupFigures } from './figures'

import './style.css'

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  setup() {
    // Tab enhancement moves SSR code blocks; wait until Vue has hydrated them.
    onMounted(setupBackendComparisons)
    onMounted(setupFigures)
  },
  enhanceApp({ app }) {
    enhanceAppWithTabs(app);
  }
}
export default Theme
