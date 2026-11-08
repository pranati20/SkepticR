/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#070a10', panel: '#0d121b', panel2: '#131a26', line: '#243146', mute: '#9ba7bb',
        disc: '#3ecf8e', luck: '#f16d7f', maybe: '#f2b84b', accent: '#5b9dff', gold: '#d4af6a',
      },
      fontFamily: { display: ['"Space Grotesk"', 'Inter', 'sans-serif'], sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'sans-serif'], mono: ['JetBrains Mono', 'ui-monospace', 'Consolas', 'monospace'] },
      keyframes: { pulseGlow: { '0%,100%': { opacity: 0.55 }, '50%': { opacity: 1 } } },
      animation: { pulseGlow: 'pulseGlow 1.4s ease-in-out infinite' },
    },
  },
  plugins: [],
}
