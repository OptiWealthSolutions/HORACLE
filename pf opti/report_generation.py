from fpdf import FPDF
import matplotlib.pyplot as plt

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 12)  # Police réduite
        self.cell(0, 8, "Optimisation de Portefeuille - Modèle de Markowitz", align='C')
        self.ln(8)

    def add_table(self, data, col_widths, row_height, col_colors):
        self.set_font("helvetica", size=9)  # Police plus petite pour les tableaux
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                if row_idx > 0 and col_idx == 1:
                    weight_value = cell.replace(' EUR', '')
                    try:
                        weight_value = float(weight_value.strip('%'))
                        if weight_value == 0:
                            self.set_text_color(255, 0, 0)
                        else:
                            self.set_text_color(0, 0, 0)
                    except ValueError:
                        pass
                else:
                    self.set_text_color(0, 0, 0)

                self.set_fill_color(*col_colors[col_idx])
                self.cell(col_widths[col_idx], row_height, str(cell), border=1, align='C', fill=True)

            self.ln(row_height)

    def add_section_title(self, title):
        self.set_font("helvetica", "B", 11)  # Police légèrement plus petite
        self.cell(0, 8, title)
        self.ln(6)

    def add_section_text(self, text):
        self.set_font("helvetica", "", 9)  # Police réduite pour le texte explicatif
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 8, text)
        self.ln(5)

    def add_pie_chart(self, image_path):
        self.image(image_path, x=45, y=self.get_y(), w=90)  # Taille réduite du graphique
        self.ln(60)

    def add_stress_test_results(self, simulation_results, expected_return, expected_volatility, min_return, max_return):
        self.add_section_title("Résultats du Test de Stress")
        table_data = [
            ["Scénario", "Rendement (%)", "Volatilité (%)"],
            ["Attendu", f"{expected_return:.2f}", f"{expected_volatility:.2f}"],
            ["Minimale", f"{min_return:.2f}", ""],
            ["Maximale", f"{max_return:.2f}", ""]
        ]

        col_widths = [70, 40, 40]
        row_height = 8  # Hauteur de ligne réduite
        col_colors = [(160, 206, 215), (136, 136, 136), (136, 136, 136)]
        self.add_table(table_data, col_widths, row_height, col_colors)
        self.ln(8)

        # Ajout de la conclusion du stress test
        vol_comment = self._generate_vol_comment(expected_volatility)
        min_return_comment = self._generate_min_return_comment(min_return)
        max_return_comment = self._generate_max_return_comment(max_return)

        self.add_section_text(vol_comment)
        self.add_section_text(min_return_comment)
        self.add_section_text(max_return_comment)

    # Méthodes séparées pour générer les commentaires afin de rendre le code plus compact
    def _generate_vol_comment(self, expected_volatility):
        if expected_volatility > 0.2:
            return "La volatilité attendue est élevée, indiquant des fluctuations substantielles sous stress."
        elif expected_volatility > 0.1:
            return "La volatilité attendue est modérée, ce qui suggère des variations raisonnables sous stress."
        else:
            return "La volatilité attendue est faible, suggérant que le portefeuille est stable sous conditions stressantes."

    def _generate_min_return_comment(self, min_return):
        if min_return < 0:
            return "Le rendement minimum est négatif, signalant des pertes possibles dans des scénarios extrêmes."
        return "Le rendement minimum reste positif, favorable même en cas de conditions de stress sévères."

    def _generate_max_return_comment(self, max_return):
        if max_return > 0:
            return "Le rendement maximum montre que des scénarios extrêmes peuvent offrir des gains significatifs."
        return "Le rendement maximum observé est inférieur au rendement attendu, limitant les gains potentiels sous stress."

def generate_pie_chart(tickers, weights, output_path="pie_chart.png"):
    plt.figure(figsize=(6, 4))  # Taille du graphique réduite
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.axis('equal')  # Pour garantir un cercle parfait
    plt.title("Pondérations des Actifs dans le Portefeuille", fontsize=10)  # Taille de titre plus petite
    plt.savefig(output_path)
    plt.close()

def generate_pdf_report(tickers, optimal_weights, optimal_return, optimal_volatility, optimal_sharpe_ratio, mean_returns, dividends, capital, 
                        allocation, total_expected_return, total_dividends, output_file, 
                        simulation_results, expected_return, expected_volatility, min_return, max_return):
    pdf = PDF()
    pdf.add_page()

    # Pondérations Optimales des Actifs
    pdf.add_section_title("Pondérations Optimales des Actifs")
    table_data = [["Actif", "Pondération"]]
    for i, ticker in enumerate(tickers):
        table_data.append([ticker, f"{optimal_weights[i] * 100:.2f}%"])
    col_widths = [50, 35]
    row_height = 8
    col_colors = [(160, 206, 215), (136, 136, 136)]
    pdf.add_table(table_data, col_widths, row_height, col_colors)

    pdf.ln(8)

    # Diagramme Circulaire
    pie_chart_path = "pie_chart.png"
    generate_pie_chart(tickers, optimal_weights, pie_chart_path)
    pdf.add_section_title("Diagramme Circulaire des Pondérations")
    pdf.add_pie_chart(pie_chart_path)

    pdf.ln(8)

    # Performances du Portefeuille
    pdf.add_section_title("Performances du Portefeuille")
    vol_comment = "La volatilité est modérée." if optimal_volatility > 0.1 else "La volatilité est faible, ce qui réduit le risque."
    sharpe_comment = "Le ratio de Sharpe est favorable."

    table_data = [
        ["Indicateur", "Valeur"],
        ["Rendement attendu du portefeuille", f"{optimal_return * 100:.2f}%"],
        ["Volatilité du portefeuille", f"{optimal_volatility * 100:.2f}%"],
        ["Ratio de Sharpe", f"{optimal_sharpe_ratio:.2f}"]
    ]
    col_widths = [80, 20]
    pdf.add_table(table_data, col_widths, row_height, col_colors)
    pdf.ln(8)
    pdf.add_section_text(vol_comment)
    pdf.add_section_text(sharpe_comment)

    # Simulation de Performance avec Capital Initial
    pdf.add_section_title("Simulation de Performance avec Capital Initial")
    pdf.add_section_text(f"Capital Initial: {capital:.2f} EUR")
    table_data = [["Actif", "Allocation (EUR)"]]
    for i, ticker in enumerate(tickers):
        table_data.append([ticker, f"{allocation[i]:.2f} EUR"])
    col_widths = [50, 30]
    pdf.add_table(table_data, col_widths, row_height, col_colors)

    pdf.ln(8)

    # Résumé de la Simulation
    pdf.add_section_title("Résumé de la Simulation")
    pdf.add_section_text(f"Rendement total attendu : {total_expected_return:.2f} EUR")
    pdf.add_section_text(f"Dividendes totaux attendus : {total_dividends:.2f} EUR")
    capital_final = capital + total_expected_return + total_dividends
    pdf.add_section_text(f"Capital final : {capital_final:.2f} EUR")

    # Ajouter les résultats du test de stress
    pdf.add_stress_test_results(simulation_results, expected_return, expected_volatility, min_return, max_return)

    pdf.output(output_file)
    print(f"Le rapport PDF a été généré : {output_file}")