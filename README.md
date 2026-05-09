# Frontier Insight

**End-to-end automated research pipeline for scientific discovery across any domain.**

Frontier Insight is a multi-agent AI system that automates the full research lifecycle — from initial brainstorming and literature synthesis to experiment design, simulation execution, cross-referencing for hidden patterns, paper writing, presentation creation, and intelligent follow-up research suggestions.

Inspired by systems like The AI Scientist but designed to be **broader, more general-purpose, and stronger at deep cross-domain synthesis**.

---

## ✨ Key Features

- **Full Research Pipeline** — Brainstorming → Literature review & cross-referencing → Experiment design → Simulation & execution → Analysis & validation → Manuscript & presentation generation → Follow-up suggestions
- **Deep Cross-Referencing Engine** — Intelligently connects literature, simulation outputs, and experimental data to surface undiscovered behaviors and novel insights
- **Multi-Agent Architecture** — Specialized agents (Researcher, Experimenter, Analyst, Writer, Reviewer) that collaborate and iterate
- **Domain Agnostic** — Works with any code-executable research (simulation, modeling, data analysis, ML experiments, etc.)
- **Reproducible & Transparent** — Full logging, artifact tracking, and versioned outputs
- **Self-Improving Loop** — Review feedback is fed back into future idea generation

---

## 🚀 Quick Start

> **Note:** This project is under active development. The following instructions will be updated as the system matures.

```bash
# Clone the repository
git clone https://github.com/jyunming/FrontierInsight.git
cd FrontierInsight

# Install dependencies
pip install -r requirements.txt

# Run the pipeline on a new research topic
python launch.py --topic "High-NA EUV stochastic effects in photoresist modeling" --output ./outputs/my-first-run
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────┐
│   Idea Generation   │  ← Brainstorming + Novelty Scoring
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Literature & Data  │  ← Semantic search + Cross-referencing
│     Cross-Ref       │    (unbury hidden patterns)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Experiment Design & │  ← Protocol + Code generation
│     Execution       │    (Python, simulators, custom tools)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Analysis &        │  ← Statistics, visualization, validation
│   Insight Synthesis │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Paper / Poster /    │  ← LaTeX, figures, slides, speech script
│   Presentation      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Follow-up Research  │  ← Prioritized next-step recommendations
└─────────────────────┘
```

---

## 📁 Project Structure

```
FrontierInsight/
├── agents/                 # Specialized agent implementations
├── core/                   # Pipeline orchestration & state management
├── templates/              # Domain-specific experiment templates
├── literature/             # Cross-referencing & embedding engine
├── generation/             # Paper, poster, presentation generators
├── evaluation/             # Self-review and quality scoring
├── examples/               # Example runs and outputs
├── docs/                   # Documentation
├── launch.py               # Pipeline entry point
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Roadmap

- [ ] Core multi-agent orchestration
- [ ] Literature + simulation cross-referencing engine
- [ ] Support for popular simulation frameworks (TorchResist, ELitho, Optolithium, etc.)
- [ ] Full LaTeX paper + figure generation
- [ ] Presentation & poster auto-generation
- [ ] Persistent research knowledge graph
- [ ] Web UI / Dashboard
- [ ] Docker + one-click deployment

---

## 🤝 Contributing

We welcome contributions! Whether you're interested in:

- Adding new agent capabilities
- Improving cross-referencing algorithms
- Adding support for new simulation tools
- Writing documentation or examples
- Suggesting new features

Please open an issue or pull request. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use Frontier Insight in your research or build upon it, please cite:

```bibtex
@software{frontier_insight_2026,
  author = {Frontier Insight Contributors},
  title  = {Frontier Insight: End-to-End Automated Research Pipeline},
  year   = {2026},
  url    = {https://github.com/jyunming/FrontierInsight}
}
```

---

## 🙏 Acknowledgments

- Inspired by [The AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI)
- Built with ideas from multi-agent frameworks (AutoGen, CrewAI, LangGraph)
- Special thanks to the open-source lithography and scientific computing communities

---

**Frontier Insight** — Pushing the frontier of automated scientific discovery.

---

*This README is a living document. Contributions and feedback are highly encouraged!*
