"""Generate a PDF evaluation report for MedRAG retrieval results."""
from __future__ import annotations
from fpdf import FPDF, XPos, YPos


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "MedRAG - Retrieval Evaluation Report", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 80, 160)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(30, 80, 160)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 6, "- " + text)
        self.ln(1)

    def table(self, headers: list, rows: list, col_widths: list, highlight_row: int = -1):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(30, 80, 160)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 8, h, border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 10)
        for r_idx, row in enumerate(rows):
            if r_idx == highlight_row:
                self.set_fill_color(220, 235, 255)
            elif r_idx % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(0, 0, 0)
            for val, w in zip(row, col_widths):
                self.cell(w, 7, str(val), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def build_report(output_path: str = "MedRAG_Evaluation_Report.pdf"):
    pdf = PDF()
    pdf.set_margins(10, 15, 10)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 60, 140)
    pdf.cell(0, 12, "MedRAG Retrieval Evaluation Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "NIH Medical Retrieval-Augmented Generation Project",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(6)

    # 1. Dataset
    pdf.section_title("1. Dataset")
    pdf.body_text(
        "The full MedQuAD dataset was downloaded from Kaggle and preprocessed before evaluation. "
        "Preprocessing steps included: null removal, short answer filtering (< 6 words), "
        "and deduplication on normalized question-answer pairs."
    )
    pdf.table(
        headers=["Metric", "Value"],
        rows=[
            ["Source", "MedQuAD (NIH Medical Q&A Pairs)"],
            ["Raw rows", "16,412"],
            ["After preprocessing", "16,350"],
            ["Rows removed", "62 (duplicates / low-quality)"],
            ["Default chunk size", "120 words"],
            ["Default chunk overlap", "20 words"],
            ["Total chunks (default)", "37,993"],
        ],
        col_widths=[90, 90],
    )

    # 2. Retrieval Method Comparison
    pdf.section_title("2. Retrieval Method Comparison (k=5)")
    pdf.body_text(
        "Each retrieval method was evaluated over all 16,350 questions. Ground truth was defined "
        "as the document_id associated with each question in the dataset. "
        "Recall@5 measures whether the correct document appeared anywhere in the top-5 retrieved chunks. "
        "Precision@5 measures what fraction of the top-5 chunks belong to the correct document."
    )
    pdf.table(
        headers=["Method", "Recall@5", "Precision@5", "Questions"],
        rows=[
            ["BM25",   "0.5871", "0.1375", "16,350"],
            ["Dense",  "0.7469", "0.1816", "16,350"],
            ["Hybrid", "0.7177", "0.1746", "16,350"],
        ],
        col_widths=[50, 45, 45, 45],
        highlight_row=1,
    )

    pdf.subsection_title("Key Findings")
    pdf.bullet(
        "Dense retrieval achieves the highest Recall@5 (0.747). Semantic embeddings capture "
        "meaning beyond exact keyword overlap, which is critical for MedQuAD since questions "
        "like 'What are the symptoms of X?' share few literal words with answer text."
    )
    pdf.bullet(
        "Hybrid retrieval underperforms Dense (0.718 vs 0.747). The BM25 candidate pool "
        "(candidate_k=20) can exclude the correct document before dense reranking sees it. "
        "Increasing candidate_k would likely close this gap."
    )
    pdf.bullet(
        "BM25 is the weakest baseline (Recall@5: 0.587). Vocabulary mismatch between short "
        "questions and long answer chunks limits keyword-based retrieval on this dataset."
    )
    pdf.bullet(
        "Precision@k is low across all methods - expected, since each question has exactly "
        "1 correct document. Precision@5 cannot exceed 0.20 even in the best case."
    )

    # 3. Chunking Analysis
    pdf.add_page()
    pdf.section_title("3. Chunking Analysis (BM25, k=5)")
    pdf.body_text(
        "BM25 was evaluated across 16 chunking configurations varying chunk size (60-200 words) "
        "and overlap (0-30 words). Each configuration re-chunks the full dataset and rebuilds "
        "the BM25 index before evaluation."
    )
    pdf.table(
        headers=["Chunk Size", "Overlap", "Chunks", "Recall@5", "Precision@5"],
        rows=[
            ["60",  "0",  "60,083",  "0.5803", "0.1398"],
            ["60",  "10", "70,285",  "0.5695", "0.1432"],
            ["60",  "20", "81,581",  "0.5490", "0.1456"],
            ["60",  "30", "100,683", "0.5280", "0.1502"],
            ["100", "0",  "39,777",  "0.6020", "0.1373"],
            ["100", "10", "42,846",  "0.5959", "0.1391"],
            ["100", "20", "44,953",  "0.5793", "0.1398"],
            ["100", "30", "47,902",  "0.5654", "0.1410"],
            ["150", "0",  "29,445",  "0.6113", "0.1341"],
            ["150", "10", "30,738",  "0.6059", "0.1348"],
            ["150", "20", "31,080",  "0.5944", "0.1345"],
            ["150", "30", "31,504",  "0.5834", "0.1341"],
            ["200", "0",  "24,999",  "0.6188", "0.1320"],
            ["200", "10", "25,555",  "0.6161", "0.1327"],
            ["200", "20", "25,359",  "0.6063", "0.1318"],
            ["200", "30", "25,149",  "0.5960", "0.1306"],
        ],
        col_widths=[34, 34, 38, 38, 40],
        highlight_row=12,
    )

    pdf.subsection_title("Key Findings")
    pdf.bullet(
        "Larger chunks consistently outperform smaller ones. Recall@5 increases from "
        "chunk_size=60 (0.580) to chunk_size=200 (0.619). Small chunks fragment sentences "
        "and split key medical terms, reducing BM25's keyword signal."
    )
    pdf.bullet(
        "Overlap hurts performance. Across every chunk size, overlap=0 produces the best "
        "Recall@5. Adding overlap increases total chunk count, introducing noise and "
        "diluting the retrieval signal."
    )
    pdf.bullet(
        "Best chunking config: chunk_size=200, overlap=0 (Recall@5=0.6188). However, even "
        "this optimal BM25 config falls well below Dense retrieval (0.747), confirming that "
        "retrieval strategy matters more than chunking for this dataset."
    )
    pdf.bullet(
        "Precision slightly favors smaller chunks with higher overlap due to repeated chunk "
        "matches, but this is misleading given Recall is the primary metric here."
    )

    # 4. Conclusion
    pdf.section_title("4. Overall Conclusion")
    pdf.table(
        headers=["Finding", "Value"],
        rows=[
            ["Best retrieval method",        "Dense (Recall@5: 0.747)"],
            ["Best chunking config (BM25)",  "chunk_size=200, overlap=0"],
            ["Biggest performance driver",   "Retrieval strategy > chunk size"],
            ["Hybrid caveat",                "Needs larger candidate_k to beat Dense"],
        ],
        col_widths=[95, 90],
    )
    pdf.body_text(
        "Dense retrieval is the recommended default for MedQuAD-style medical QA. "
        "Hybrid retrieval is a strong alternative if the BM25 candidate pool is made larger. "
        "For chunking, larger non-overlapping chunks work best with BM25, while dense retrieval "
        "is more robust to chunk size variation due to its semantic matching capability."
    )

    pdf.output(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    build_report()
