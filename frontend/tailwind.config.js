/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#080c14',
        surface: '#0d1117',
        border: '#1a2234',
        primary: '#0ea5e9',
        danger: '#ef4444',
        success: '#10b981',
        warning: '#f59e0b',
        muted: '#475569',
        text: '#f1f5f9',
        textMuted: '#94a3b8'
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
