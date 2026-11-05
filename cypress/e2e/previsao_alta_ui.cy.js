
// Teste do dashboard Streamlit: página "Probabilidade de alta" (issue #13).
// Pré-requisito: dashboard no ar em http://localhost:8501
// streamlit run dashboard/app.py
// Funciona com ou sem a API (sem ela, o dashboard usa o modelo local).

describe("Dashboard - probabilidade de alta", () => {
  beforeEach(() => {
    cy.visit("/");

    // Abre a página pelo menu lateral
    cy.get('[data-testid="stSidebar"]')
      .contains("Probabilidade de alta")
      .click();

    cy.contains("Quando o paciente pode ir embora?")
      .should("be.visible");
  });

  it("mostra a estimativa e a faixa de incerteza", () => {
    cy.contains("button", "Probabilidade de Alta").click();

    cy.contains("Estimativa").should("be.visible");
    cy.contains(/\d+(\.\d+)? dias/).should("be.visible");

    // Verifica o texto atual da faixa
    cy.contains(/Faixa estimada:\s*entre/i)
      .should("be.visible");
  });

  it("a faixa mostrada tem mínimo menor ou igual ao máximo", () => {
    cy.contains("button", "Probabilidade de Alta").click();

    cy.contains(/Faixa estimada:\s*entre/i)
      .invoke("text")
      .then((texto) => {
        const m = texto.match(
          /entre\s+([\d.,]+)\s+e\s+([\d.,]+)\s+dias/i
        );

        expect(m, "texto da faixa").to.not.be.null;

        const minimo = Number(m[1].replace(",", "."));
        const maximo = Number(m[2].replace(",", "."));

        expect(minimo).to.be.at.most(maximo);
      });
  });

  it("mostra o aviso de que o médico decide", () => {
    cy.contains("button", "Probabilidade de Alta").click();

    cy.contains("O médico decide de verdade")
      .should("be.visible");
  });
});
