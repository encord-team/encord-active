/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {},
  },
  daisyui: {
    themes: [
      {
        mytheme: {
          primary: "#a991f7",
          secondary: "#f6d860",
          accent: "#37cdbe",
          neutral: "#D7DDE8",
          "base-100": "#ffffff",
        },
      },
    ],
  },
  plugins: [require("@tailwindcss/line-clamp"), require("daisyui")],
};
