const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    baseUrl: "http://localhost:8501",
    specPattern: "cypress/e2e/**/*.cy.js",
    supportFile: false,
  },

  expose: {
    apiUrl: "http://localhost:8000/api/v1",
  },
});