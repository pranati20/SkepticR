/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b1220', panel: '#101a2e', panel2: '#152038', line: '#22304d', mute: '#a9b4c7',
        disc: '#34c38f', luck: '#e5677a', maybe: '#e8b04b', accent: '#3b82f6', gold: '#c9a45c',
      },
      fontFamily: { display: ['"Space Grotesk"', 'Inter', 'sans-serif'], sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'sans-serif'], mono: ['JetBrains Mono', 'ui-monospace', 'Consolas', 'monospace'] },
      keyframes: { pulseGlow: { '0%,100%': { opacity: 0.55 }, '50%': { opacity: 1 } } },
      animation: { pulseGlow: 'pulseGlow 1.4s ease-in-out infinite' },
    },
  },
  plugins: [],
}
