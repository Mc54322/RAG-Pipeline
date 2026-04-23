"""
Create benchmark PDF documents for the RAG pipeline evaluation.

Generates five topic-specific PDFs in data/benchmark/, each containing
a focused passage on a distinct subject. These are the canonical source
documents used by scripts/run_benchmark.py - editing them changes what
the benchmark retrieves and evaluates against.

Usage:
    python scripts/create_benchmark_data.py

Requires PyMuPDF (fitz), already in requirements.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Topic content
# Each entry is (filename, title, body_paragraphs).
# Keep content tight and keyword-rich so relevance checks stay reliable.
# ---------------------------------------------------------------------------

TOPICS: list[tuple[str, str, list[str]]] = [
    (
        "machine_learning.pdf",
        "Machine Learning and Neural Networks",
        [
            "Machine learning is a subset of artificial intelligence where algorithms "
            "learn patterns from data without being explicitly programmed. Rather than "
            "following hand-written rules, the model adjusts its internal parameters "
            "through exposure to examples, improving its predictions over time.",

            "The primary optimisation algorithm used to train neural networks is gradient "
            "descent. At each training step, the algorithm computes the gradient of a loss "
            "function with respect to the model weights and nudges those weights in the "
            "direction that reduces the loss. Variants such as stochastic gradient descent "
            "and Adam make this process more efficient for large datasets.",

            "Common neural network architectures include convolutional neural networks "
            "(CNNs) for image recognition, recurrent neural networks (RNNs) for sequential "
            "data, and transformers for natural language processing tasks. Transformers "
            "introduced the self-attention mechanism, which allows each token in a sequence "
            "to attend to every other token simultaneously, enabling highly parallelisable "
            "training on modern hardware.",

            "Regularisation techniques such as dropout and weight decay prevent overfitting "
            "by discouraging the model from relying too heavily on any single feature. "
            "Batch normalisation stabilises training by normalising activations within each "
            "mini-batch, allowing higher learning rates and faster convergence.",
        ],
    ),
    (
        "climate_change.pdf",
        "Climate Change and Global Warming",
        [
            "Climate change refers to long-term shifts in global temperatures and weather "
            "patterns. While natural factors such as volcanic eruptions and solar variation "
            "have always influenced the climate, human activities since the industrial "
            "revolution have become the dominant driver of observed warming.",

            "The primary cause of current climate change is the burning of fossil fuels - "
            "coal, oil, and natural gas - which releases carbon dioxide and other greenhouse "
            "gases into the atmosphere. These gases trap outgoing infrared radiation, raising "
            "the average surface temperature of the Earth. Methane from agriculture and "
            "nitrous oxide from fertilisers are also significant contributors.",

            "Consequences of global warming include rising sea levels from melting ice "
            "sheets and glaciers, more frequent and intense extreme weather events such as "
            "heatwaves and hurricanes, and widespread disruption to ecosystems. Ocean "
            "acidification - caused by the ocean absorbing excess carbon dioxide - "
            "threatens coral reefs and marine biodiversity.",

            "Mitigation strategies include transitioning to renewable energy sources such "
            "as solar and wind power, improving energy efficiency in buildings and transport, "
            "reforesting degraded land, and developing carbon capture technologies. "
            "International agreements such as the Paris Agreement set targets for limiting "
            "warming to 1.5 °C above pre-industrial levels.",
        ],
    ),
    (
        "french_revolution.pdf",
        "The French Revolution",
        [
            "The French Revolution began in 1789 and fundamentally transformed France's "
            "political and social order. The revolution dismantled the absolute monarchy "
            "that had ruled France for centuries and replaced it with a republic founded "
            "on the principles of liberty, equality, and fraternity.",

            "Root causes of the revolution included a severe fiscal crisis caused by "
            "France's costly involvement in the American Revolutionary War, deep social "
            "inequality between the privileged estates (clergy and nobility) and the "
            "commoners, repeated harvest failures that drove up bread prices, and the "
            "spread of Enlightenment ideas about individual rights and popular sovereignty.",

            "The storming of the Bastille on 14 July 1789 became the defining symbol "
            "of the uprising. The Bastille was a royal fortress and prison whose fall "
            "represented the collapse of royal authority. Bastille Day is still "
            "commemorated as France's national holiday. The revolution subsequently "
            "abolished feudalism, issued the Declaration of the Rights of Man and "
            "of the Citizen, and eventually led to the execution of King Louis XVI.",

            "The revolution entered a radical phase known as the Reign of Terror "
            "(1793–1794), during which the Committee of Public Safety, led by "
            "Maximilien Robespierre, oversaw the execution of thousands of perceived "
            "enemies of the republic. The Terror ended with Robespierre's own arrest "
            "and execution, paving the way for the more moderate Directory government "
            "and eventually Napoleon Bonaparte's rise to power.",
        ],
    ),
    (
        "dna_genetics.pdf",
        "DNA and Genetics",
        [
            "DNA, or deoxyribonucleic acid, is the molecule that carries the genetic "
            "instructions for the development, functioning, growth, and reproduction of "
            "all known living organisms and many viruses. It is composed of four "
            "nucleotide bases - adenine, thymine, cytosine, and guanine - arranged "
            "along a sugar-phosphate backbone.",

            "The double helix structure of DNA was described by James Watson and "
            "Francis Crick in 1953, based in part on X-ray crystallography work by "
            "Rosalind Franklin and Maurice Wilkins. In the double helix, two strands "
            "wind around each other, with complementary base pairs (A–T and C–G) "
            "held together by hydrogen bonds. This complementary pairing enables "
            "accurate replication during cell division.",

            "Genes are specific sequences of DNA that encode instructions for building "
            "proteins. The process of gene expression involves transcription (copying "
            "DNA into messenger RNA) and translation (using the mRNA sequence to "
            "assemble a protein from amino acids). Mutations - changes in the DNA "
            "sequence - can alter protein function and may give rise to hereditary "
            "diseases or drive evolutionary change over generations.",

            "Modern genomics allows scientists to sequence entire genomes rapidly and "
            "cheaply. Technologies such as CRISPR-Cas9 enable precise editing of "
            "specific genes, opening possibilities for treating genetic diseases. "
            "The Human Genome Project, completed in 2003, provided the first "
            "complete reference sequence of the human genome - approximately three "
            "billion base pairs encoding around 20,000 protein-coding genes.",
        ],
    ),
    (
        "python_programming.pdf",
        "Python Programming Language",
        [
            "Python was created by Guido van Rossum and first released in 1991. "
            "Its design philosophy emphasises code readability and clean syntax, "
            "enforced through significant indentation rather than braces or keywords "
            "to delimit blocks. This makes Python programs easier to read and "
            "maintain compared to many other languages.",

            "Python's extensive ecosystem of libraries has made it the dominant "
            "language for data science and machine learning research. NumPy provides "
            "efficient array operations, Pandas enables tabular data manipulation, "
            "Matplotlib and Seaborn cover visualisation, and deep learning frameworks "
            "such as PyTorch and TensorFlow allow researchers to build and train "
            "neural networks with relatively little boilerplate code.",

            "Python's interpreter and dynamic typing allow rapid prototyping - code "
            "can be written and tested interactively in a REPL or Jupyter notebook. "
            "However, Python's Global Interpreter Lock (GIL) limits true parallelism "
            "in CPU-bound multi-threaded programs; CPU-intensive tasks are typically "
            "offloaded to C extensions (as NumPy and PyTorch do internally) or handled "
            "with the multiprocessing module.",

            "The Python Software Foundation (PSF) oversees the language's development "
            "and publishes a new major version roughly every year. Python 3 introduced "
            "breaking changes from Python 2 (such as making print a function and "
            "requiring explicit encoding declarations), and Python 2 reached end of "
            "life in January 2020. Modern Python versions include pattern matching, "
            "structural subtyping via protocols, and a per-interpreter GIL option.",
        ],
    ),
]


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def create_pdf(output_path: Path, title: str, paragraphs: list[str]) -> None:
    """
    Write a single-topic PDF to output_path.

    Layout: title on the first line, then each paragraph separated by a
    blank line. Uses PyMuPDF's insert_textbox for automatic line wrapping
    within the page margins.

    Args:
        output_path: Destination file path (created or overwritten).
        title: Document title, rendered in larger text at the top.
        paragraphs: Body paragraphs, each separated by a blank line.
    """
    doc = fitz.open()  # new empty document
    page = doc.new_page(width=595, height=842)  # A4

    margin = 60
    y = margin

    # Title
    page.insert_text(
        (margin, y),
        title,
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )
    y += 30

    # Separator line
    page.draw_line((margin, y), (595 - margin, y), color=(0.5, 0.5, 0.5), width=0.5)
    y += 16

    # Body paragraphs
    text_rect = fitz.Rect(margin, y, 595 - margin, 842 - margin)
    body_text = "\n\n".join(paragraphs)
    page.insert_textbox(
        text_rect,
        body_text,
        fontsize=11,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_LEFT,
    )

    doc.save(str(output_path))
    doc.close()


def main() -> None:
    output_dir = Path(__file__).parent.parent / "data" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing benchmark PDFs to {output_dir}/\n")

    for filename, title, paragraphs in TOPICS:
        path = output_dir / filename
        create_pdf(path, title, paragraphs)
        size_kb = path.stat().st_size / 1024
        print(f"  {filename:<35}  {size_kb:.1f} KB")

    print(f"\nDone - {len(TOPICS)} PDFs created.")
    print("Run 'python scripts/run_benchmark.py' to evaluate retrieval quality.")


if __name__ == "__main__":
    main()
