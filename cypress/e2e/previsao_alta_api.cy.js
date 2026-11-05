// Testes da API de previsão de alta (issue #13: faixa de incerteza).
// Pré-requisito: API no ar em http://localhost:8000 (backend/run.sh).

describe("API - previsão de alta", () => {
  const url = () => `${Cypress.expose("apiUrl")}/previsao/alta`;

  const paciente = (extra = {}) => ({
    idade: 55,
    dias_internado: 3,
    complexidade: 3,
    setor: "UTI",
    comorbidades: 1,
    ...extra,
  });

  it("API está no ar", () => {
  cy.request("http://localhost:8000/health")
    .its("body")
    .its("status")
    .should("eq", "ok");
});

  it("devolve estimativa com faixa de incerteza coerente", () => {
    cy.request("POST", url(), paciente()).then(({ status, body }) => {
      expect(status).to.eq(200);
      expect(body.dias_restantes).to.be.a("number").and.to.be.gte(0.5);
      expect(body.data_prevista).to.match(/^\d{4}-\d{2}-\d{2}$/);

      // faixa presente e ordenada
      expect(body.intervalo_min).to.be.a("number");
      expect(body.intervalo_max).to.be.a("number");
      expect(body.intervalo_min).to.be.at.least(0.5);
      expect(body.intervalo_min).to.be.at.most(body.dias_restantes);
      expect(body.intervalo_max).to.be.at.least(body.dias_restantes);

      // confiança = cobertura real da faixa (0.8), não um valor inventado
      expect(body.confianca).to.eq(0.8);
    });
  });

  it("faixa da UTI é mais larga que a da Pediatria", () => {
    cy.request("POST", url(), paciente({ setor: "UTI" })).then((uti) => {
      cy.request(
        "POST",
        url(),
        paciente({ setor: "Pediatria", idade: 8, dias_internado: 2 })
      ).then((ped) => {
        const larguraUti = uti.body.intervalo_max - uti.body.intervalo_min;
        const larguraPed = ped.body.intervalo_max - ped.body.intervalo_min;
        expect(larguraUti).to.be.greaterThan(larguraPed);
      });
    });
  });

  it("aceita todos os setores", () => {
    ["UTI", "Enfermaria", "Emergência", "Cirúrgico", "Pediatria"].forEach(
      (setor) => {
        const idade = setor === "Pediatria" ? 8 : 55;
        cy.request("POST", url(), paciente({ setor, idade }))
          .its("status")
          .should("eq", 200);
      }
    );
  });

  it("rejeita setor desconhecido", () => {
    cy.request({
      method: "POST",
      url: url(),
      body: paciente({ setor: "Oncologia" }),
      failOnStatusCode: false,
    })
      .its("status")
      .should("eq", 422);
  });

  it("rejeita complexidade fora de 1 a 5", () => {
    cy.request({
      method: "POST",
      url: url(),
      body: paciente({ complexidade: 9 }),
      failOnStatusCode: false,
    })
      .its("status")
      .should("eq", 422);
  });
});
