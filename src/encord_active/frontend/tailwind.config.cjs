/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        gray: {
          2: "#FAFAFA",
          8: "#595959",
          9: "#434343",
          10: "#262626",
        },
        warning: "#ec9c27",
        severe: "#cf1322",
        healthy: "#3f8600",
      },
    },
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
  plugins: [
    require("@tailwindcss/line-clamp"),
    require("daisyui"),
    ({ addVariant }) => {
      addVariant("not-first", "& > *:not(:first-child)");
      addVariant("not-last", "& > *:not(:last-child)");
    },
  ],
};
