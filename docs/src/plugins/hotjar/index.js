module.exports = function (_, options) {
  const { applicationId } = options;

  if (!applicationId) {
    throw new Error(
      `You need to pass 'applicationId' as options to the plugin`
    );
  }

  return {
    name: "docusaurus-hotjar-plugin",
    injectHtmlTags: () =>
      !process.env.NODE_ENV === "production"
        ? {}
        : {
            headTags: [
              {
                attributes: { type: "module" },
                tagName: "script",
                innerHTML: `
                  (function (h, o, t, j, a, r) {
                    h.hj =
                      h.hj ||
                      function () {
                        (h.hj.q = h.hj.q || []).push(arguments);
                      };
                    h._hjSettings = { hjid: ${applicationId}, hjsv: 6 };
                    a = o.getElementsByTagName("head")[0];
                    r = o.createElement("script");
                    r.async = 1;
                    r.src = t + h._hjSettings.hjid + j + h._hjSettings.hjsv;
                    a.appendChild(r);
                  })(window, window.document, "https://static.hotjar.com/c/hotjar-", ".js?sv=");
                `,
              },
            ],
          },
  };
};
