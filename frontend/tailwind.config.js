/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#080b12', panel: '#0f1520', panel2: '#141c2b', line: '#1e2a3d', mute: '#7d8aa3',
        disc: '#34d399', luck: '#fb7185', maybe: '#fbbf24', accent: '#38bdf8',
      },
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'sans-serif'], mono: ['JetBrains Mono', 'ui-monospace', 'Consolas', 'monospace'] },
      keyframes: { pulseGlow: { '0%,100%': { opacity: 0.55 }, '50%': { opacity: 1 } } },
      animation: { pulseGlow: 'pulseGlow 1.4s ease-in-out infinite' },
    },
  },
  plugins: [],
}
